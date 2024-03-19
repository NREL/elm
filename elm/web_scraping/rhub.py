"""
Code to build Corpus from the researcher hub.
"""
import os
import os.path
import logging
from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup
import openai
from rex import init_logger

from elm.embed import ChunkAndEmbed

# initialize logger
logger = logging.getLogger(__name__)
init_logger(__name__, log_level='DEBUG')
init_logger('elm', log_level='INFO')

# set openAI variables
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'

ChunkAndEmbed.EMBEDDING_MODEL = 'text-embedding-ada-002-2'
ChunkAndEmbed.EMBEDDING_URL = ('https://stratus-embeddings-south-central.'
                               'openai.azure.com/openai/deployments/'
                               'text-embedding-ada-002-2/embeddings?'
                               f'api-version={openai.api_version}')
ChunkAndEmbed.HEADERS = {"Content-Type": "application/json",
                         "Authorization": f"Bearer {openai.api_key}",
                         "api-key": f"{openai.api_key}"}

class ResearchOutputs():
    """Class to handle publications portion of the NREL researcher hub."""
    BASE_URL = "https://research-hub.nrel.gov/en/publications/?page=0"


    def __init__(self, url, n_pages=1, txt_dir='./ew_txt'):

        self.text_dir = txt_dir
        self.all_links = []
        for p in range(0, n_pages):
            url = url + f"?page={p}"
            page = urlopen(url)
            html = page.read().decode("utf-8")
            self.soup = BeautifulSoup(html, "html.parser")

            self.target = self.soup.find('ul', {'class': 'list-results'})
            self.docs = self.target.find_all('a', {'class': 'link'})

            page_links = [d['href'] for d in self.docs if 
                          '/publications/' in d['href']]
            self.all_links.extend(page_links)

    def _scrape_authors(self, soup_inst):
        """Scrape the names of authors associated with given publication.

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        author names (str): all authors that contributed to publication
        """

        authors = soup_inst.find('p', {'class': 'relations persons'}).text

        return authors

    def _scrape_links(self, soup_inst):
        """Scrape the links under 'Access to Document' header 
        for a publication.

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        doi link (str)
        pdf link (str): link to pdf if it exists
        """

        doi_target = soup_inst.find('ul', {'class': 'dois'})
        if doi_target:
            doi = doi_target.find('a')['href']
        else:
            doi = ''

        pdf_target = soup_inst.find('ul', {'class': 'links'})
        if pdf_target:
            pdf = pdf_target.find('a')['href']
        else:
            pdf = ''

        return doi, pdf

    def _scrape_category(self, soup_inst):
        """Scrape the category (ex: Technical Report, Journal Article, etc)
        for a given publication.

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        category (str)
        """

        category = soup_inst.find('span', 
                                  {'class': 'type_classification'}).text

        return category

    def _scrape_year(self, soup_inst):
        """Scrape publication year for a given publication.

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        year (str): the year a record was published
        """
        year = soup_inst.find('span', {'class': 'date'}).text

        return year

    def _scrape_id(self, soup_inst):
        """Scrape the NREL Publication Number for a given publication.

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        NREL Publication Number (str)
        """

        nrel_id = soup_inst.find('ul', {'class': 'relations keywords'}).text

        return nrel_id

    def build_meta(self):
        """Build a meta dataframe containing relevant information
         for publications.

        Returns
        -------
        pd.DataFrame
        """
        publications_meta = pd.DataFrame(columns=('title', 'nrel_id',
                                                    'authors','year',
                                                    'url', 'fn', 'doi',
                                                    'pdf_url', 'category'))
        for link in self.all_links[:10]:  # quantity control here #
            page = urlopen(link)
            html = page.read().decode("utf-8")
            meta_soup = BeautifulSoup(html, "html.parser")

            title = meta_soup.find('h1').text
            nrel_id = self._scrape_id(meta_soup)
            fn = os.path.basename(link) + '.txt'
            authors = self._scrape_authors(meta_soup)
            doi = self._scrape_links(meta_soup)[0]
            pdf_url = self._scrape_links(meta_soup)[1]
            category = self._scrape_category(meta_soup)
            year = self._scrape_year(meta_soup)

            new_row = {'title': title,
                       'nrel_id': nrel_id,
                        'year': year,
                        'authors': authors,
                        'url': link,
                        'fn': fn,
                        'doi': doi,
                        'pdf_url': pdf_url,
                        'category': category
            }

            publications_meta.loc[len(publications_meta)] = new_row

        return publications_meta

    # Scrape Abstracts for associated projects
    def scrape_abstracts(self, out_dir):
        """
        Description
        ----------
        Scrapes abstract for each publication listed.

        Parameters
        ----------
        out_dir: str
            Directory where the .txt files should be saved.

        Returns
        ---------
        Text file containing abstract
        """
        os.makedirs(out_dir, exist_ok=True)
        url_list = self.all_links[:10]  # quantity control here #

        for i, pub in enumerate(url_list):
            fn = os.path.basename(pub) + '.txt'
            out_fp = os.path.join(out_dir, fn)
            if not os.path.exists(out_fp):
                page = urlopen(pub)
                html = page.read().decode("utf-8")
                soup = BeautifulSoup(html, "html.parser")

                title = soup.find('h1').text
                target = soup.find('h2', string='Abstract')
                if target:
                    abstract = target.find_next_siblings()[0].text
                    full_txt = (f'The report titled {title} can be '
                                f'summarized as follows: {abstract}')
                    with open(out_fp, "w") as text_file:
                        text_file.write(full_txt)
                    logger.info('Processing {}/{}: {}'.format(i + 1,
                                                              len(url_list),
                                                              pub))
                else:
                    logger.info('Abstract not found for {}'.format(pub))
            else:
                logger.info('{} has already been processed.'.format(out_fp))

        return logger.info('Finished processing abstracts')


class ResearcherProfiles():
    """
    Class to handle researcher profiles portion of the NREL researcher hub.
    """
    BASE_URL = "https://research-hub.nrel.gov/en/persons/?page=0"

    def __init__(self, url, n_pages=1, txt_dir='./ew_txt'):
        self.text_dir = txt_dir
        self.profile_links = []
        for p in range(0, n_pages):
            url_base = url + f"?page={p}"
            page = urlopen(url_base)
            html = page.read().decode("utf-8")
            soup = BeautifulSoup(html, "html.parser")

            target = soup.find('ul', {'class': 'grid-results'})
            docs = target.find_all('a', {'class': 'link'})

            page_links = [d['href'] for d in docs if '/persons/' in d['href']]
            self.profile_links.extend(page_links)

    def build_meta(self):
        """Build a meta dataframe containing relevant information for
        researchers.

        Returns
        -------
        pd.DataFrame
        """
        url_list = self.profile_links
        profiles_meta = pd.DataFrame(columns=('title', 'nrel_id',
                                                'email', 'url', 'fn',
                                                'category'))
        for link in url_list[:10]:  # quantity control here #
            page = urlopen(link)
            html = page.read().decode("utf-8")
            meta_soup = BeautifulSoup(html, "html.parser")

            title = meta_soup.find('h1').text
            email_target = meta_soup.find('a', {'class': 'email'})
            if email_target:
                email = meta_soup.find('a',
                                       {'class': 'email'}
                                       ).text.replace('nrelgov','@nrel.gov')
            else:
                email = ''
            id = os.path.basename(link)
            fn = os.path.basename(link) + '.txt'

            new_row = {'title': title,
                       'nrel_id': id,
                       'email': email,
                        'url': link,
                        'fn': fn,
                        'category': 'Researcher Profile'
            }

            profiles_meta.loc[len(profiles_meta)] = new_row

        return profiles_meta

    def _scrape_title(self, soup_inst):
        """
        Description
        ----------
        Scrapes name and position for each researcher.

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given researcher.

        Returns
        -------
        intro (str): string containing researchers name and position.
        """

        r = soup_inst.find('h1').text

        if soup_inst.find('span', {'class': 'job-title'}):
            j = soup_inst.find('span', {'class': 'job-title'}).text
            intro = (f'The following is brief biography for {r} '
                    f'who is a {j} at the National Renewable Energy Laboratory:\n')
        else:
            intro = (f'The following is brief biography for {r}'
                     f'who works for the National Renewable Energy Laboratory:\n')

        return intro

    def _scrape_bio(self, soup_inst):
        """
        Description
        ----------
        Scrapes 'Personal Profile' section for each researcher.

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given researcher.

        Returns
        -------
        bio (str): string containing text from profile.
        """
        target = soup_inst.find('h3', string="Personal Profile")

        bio = ''
        if target:
            for sib in target.find_next_siblings():
                if sib.name == "h3":
                    break
                bio = bio + sib.text

        return bio

    def _scrape_lists(self, soup_inst, heading):
        """
        Description
        ----------
        Scrapes sections such as 'Professional Experience' and
        'Research Interests'

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given researcher.
        heading: str
            Section to scrape. Should be 'Professional Experience' or
            'Research Interests'

        Returns
        -------
        text (str): string containing contents from the provided section.
        """
        r = soup_inst.find('h1').text
        target = soup_inst.find('h3', string=heading)

        exp_list = []

        if target:
            for sib in target.find_next_siblings():
                exp_list.append(sib.text)

            exp = ', '.join(exp_list)

            text = f"{r}'s {heading} includes the following:\n{exp} "
        else:
            text = ''

        return text

    def _scrape_education(self, soup_inst):
        """
        Description
        ----------
        Scrapes and reformats 'Education/Academic Qualification'
        section for each researcher.

        Parameters
        ----------
        soup_inst : obj
            Active beautiful soup instance for the url associated with a
            given researcher.

        Returns
        -------
        full_text (str): string containing researchers education
        (level, focus, and institution).
        """
        r = soup_inst.find('h1').text
        target = soup_inst.find('h3', string='Education/Academic Qualification')

        full_text = ''
        if target:
            for sib in target.find_next_siblings():
                t = sib.text
                if len(t.split(',')) == 3:
                    level = t.split(',')[0]
                    deg = t.split(',')[1]
                    inst = t.split(',')[2]

                    text = f"{r} received a {level} degree in {deg} from the {inst}. "
                elif len(t.split(',')) == 2:
                    level = t.split(',')[0]
                    inst = t.split(',')[1]

                    text = f"{r} received a {level} degree from the {inst}. "

                full_text = full_text + text

        return full_text

    def _scrape_publications(self, profile_url):
        """
        Description
        ----------
        Scrapes the name of each publication that a researcher contributed to.

        Parameters
        ----------
        profile_url : str
            Link to a specific researchers profile.

        Returns
        -------
        text (str): string containing all publications for a given researcher.
        """
        pubs_url = profile_url + '/publications/'
        page = urlopen(pubs_url)
        html = page.read().decode("utf-8")
        pubs_soup = BeautifulSoup(html, "html.parser")

        r = pubs_soup.find('h1').text
        target = pubs_soup.find_all('h3', {'class': 'title'})

        pubs = []
        if target:
            for p in target:
                pubs.append(p.text)

            pubs = ', '.join(pubs)
            text = f'{r} has contributed to the following publications: {pubs}.'
        else:
            text = ''

        return text

    def _scrape_similar(self, profile_url):
        """
        Description
        ----------
        Scrapes the names listed under the 'Similar Profiles' section.

        Parameters
        ----------
        profile_url : str
            Link to a specific researchers profile.

        Returns
        -------
        text (str): string containing similar researchers.
        """
        sim_url = profile_url + '/similar/'
        sim_page = urlopen(sim_url)
        sim_html = sim_page.read().decode("utf-8")
        sim_soup = BeautifulSoup(sim_html, "html.parser")

        r = sim_soup.find('h1').text
        target = sim_soup.find_all('h3', {'class': 'title'})

        similar = []
        if target:
            for p in target:
                similar.append(p.text)

            similar = ', '.join(similar)
            text = f'{r} has worked on projects with {similar}.'
        else:
            text = ''

        return text

    def scrape_profiles(self, out_dir):
        """
        Description
        ----------
        Scrapes profiles for each researcher.

        Parameters
        ----------
        out_dir: str
            Directory where the .txt files should be saved.

        Returns
        ---------
        Text file containing information from the profile.
        """
        os.makedirs(out_dir, exist_ok=True)
        url_list = self.profile_links[:10]  # quantity control here #

        for i, prof in enumerate(url_list):
            f = os.path.basename(prof) + '.txt'
            txt_fp = os.path.join(out_dir, f)
            if not os.path.exists(txt_fp):
                page = urlopen(prof)
                html = page.read().decode("utf-8")
                prof_soup = BeautifulSoup(html, "html.parser")

                r = prof_soup.find('h1').text

                intro = self._scrape_title(prof_soup)
                bio = self._scrape_bio(prof_soup)
                exp = self._scrape_lists(prof_soup, 'Professional Experience')
                interests = self._scrape_lists(prof_soup, 'Research Interests')
                edu = self._scrape_education(prof_soup)
                pubs = self._scrape_publications(prof)
                similar = self._scrape_similar(prof)

                full_txt = (intro + bio + '\n' + exp + '\n' +
                        interests + '\n' + edu + '\n' +
                        pubs + '\n' + similar)

                with open(txt_fp, "w") as text_file:
                    text_file.write(full_txt)
                logger.info('Profile {}/{}: {} saved to {}'.format(i + 1, 
                                                                   len(url_list),
                                                                   r, txt_fp))

            else:
                logger.info('Profile {}/{} already exists.'.format(i+1, 
                                                                   len(url_list)))
        return logger.info('Finished processing profiles')
