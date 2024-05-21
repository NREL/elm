"""
Code to build Corpus from the researcher hub.
"""
import os
import os.path
import logging
from urllib.request import urlopen
import json
import math
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ResearchOutputs():
    """Class to handle publications portion of the NREL researcher hub."""
    BASE_URL = "https://research-hub.nrel.gov/en/publications/?page=0"

    def __init__(self, url, n_pages=1, txt_dir='./ew_txt'):
        """
        Parameters
        ----------
        url : str
            Research hub publications URL, most likely
            https://research-hub.nrel.gov/en/publications/
        n_pages : int
            Number of pages to get from the API. Typical response has 50
            entries per page. Default of 1 ensures that this class doesnt hang
            on a million responses.
        txt_dir : str
            File directory where you would like to save output .txt files.
        """

        self.text_dir = txt_dir
        self.all_links = []
        for p in range(0, n_pages):
            url = url + f"?page={p}"
            html = self.html_response(url)
            self.soup = BeautifulSoup(html, "html.parser")

            self.target = self.soup.find('ul', {'class': 'list-results'})
            self.docs = self.target.find_all('a', {'class': 'link'})

            page_links = [d['href'] for d in self.docs if
                          '/publications/' in d['href']]
            self.all_links.extend(page_links)

    def html_response(self, url):
        """Function to retrieve html response.

        Parameters
        ----------
        url : str
            URL of interest.

        Returns
        -------
        html : str
            HTML response output.
        """
        with urlopen(url) as page:
            html = page.read().decode("utf-8")

        return html

    def _scrape_authors(self, soup_inst):
        """Scrape the names of authors associated with given publication.

        Parameters
        ----------
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        authors : list
            List of all authors (strings) that contributed to publication.
        """

        authors = soup_inst.find('p', {'class': 'relations persons'}).text

        return authors

    def _scrape_links(self, soup_inst):
        """Scrape the links under 'Access to Document' header
        for a publication.

        Parameters
        ----------
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        doi link : str
            DOI link for a reference if it exists.
        pdf link : str
            PDF link for a reference if it exists
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
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        category : str
            Publication category for a given record.
        """

        try:
            category = soup_inst.find('span',
                                      {'class':
                                       'type_classification'}).text
        except AttributeError:
            category = soup_inst.find('span',
                                      {'class':
                                       'type_classification_parent'}).text

        return category

    def _scrape_year(self, soup_inst):
        """Scrape publication year for a given publication.

        Parameters
        ----------
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        year : str
            The year a record was published.
        """
        year = soup_inst.find('span', {'class': 'date'}).text

        return year

    def _scrape_id(self, soup_inst):
        """Scrape the NREL Publication Number for a given publication.

        Parameters
        ----------
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given publication.

        Returns
        -------
        NREL Publication Number: str
            Publication number for a record, unique identifier.
        """

        nrel_id = soup_inst.find('ul', {'class': 'relations keywords'}).text

        return nrel_id

    def build_meta(self):
        """Build a meta dataframe containing relevant information
         for publications.

        Returns
        -------
        publications_meta : pd.DataFrame
            Dataframe containing metadata for publications.
        """
        publications_meta = pd.DataFrame(columns=('title', 'nrel_id',
                                                  'authors', 'year',
                                                  'url', 'doi',
                                                  'pdf_url', 'category'))
        for link in self.all_links[:20]:  # quantity control here #
            with urlopen(link) as page:
                html = page.read().decode("utf-8")
            meta_soup = BeautifulSoup(html, "html.parser")

            title = meta_soup.find('h1').text
            nrel_id = self._scrape_id(meta_soup)
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
                       'doi': doi,
                       'pdf_url': pdf_url,
                       'category': category
                       }

            publications_meta.loc[len(publications_meta)] = new_row

        return publications_meta

    def download_pdf(self, pdf_dir, txt_dir, soup_inst):
        """Downloads a pdf for a given link

        Parameters
        ----------
        out_dir: str
            Directory where the .pdf files should be saved.
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance used to locate pdf url.
        """
        pdf_target = soup_inst.find('ul', {'class': 'links'})
        if pdf_target:
            pdf_url = pdf_target.find('a')['href']

        fn = os.path.basename(pdf_url)
        fp_out = os.path.join(pdf_dir, fn)

        if pdf_url and pdf_url.endswith('.pdf'):
            if not os.path.exists(fp_out):
                session = requests.Session()
                response = session.get(pdf_url)
                with open(fp_out, 'wb') as f_pdf:
                    f_pdf.write(response.content)
                logger.info('Downloaded {}'.format(fn))
            else:
                logger.info('{} has already been downloaded'.format(fn))
        elif not pdf_url.endswith('.pdf'):
            parent_url = soup_inst.find(property="og:url")['content']
            fn = os.path.basename(parent_url) + '_abstract.txt'
            logger.info('No PDF file for {}. Processing abstract.'.format(fn))
            self.scrape_abstract(txt_dir, fn, soup_inst)

    def scrape_abstract(self, out_dir, fn, soup_inst):
        """Scrapes abstract for a provided publication

        Parameters
        ----------
        out_dir: str
            Directory where the .txt files should be saved.
        fn: str
            File name for saving the file.
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance used for scraping.
        """
        out_fp = os.path.join(out_dir, fn)
        if not os.path.exists(out_fp):
            title = soup_inst.find('h1').text
            target = soup_inst.find('h2', string='Abstract')
            if target:
                abstract = target.find_next_siblings()[0].text
                full_txt = (f'The report titled {title} can be '
                            f'summarized as follows: {abstract}')
                with open(out_fp, "w") as text_file:
                    text_file.write(full_txt)
            else:
                logger.info('Abstract not found for {}'.format(fn))
        else:
            logger.info('{} has already been processed.'.format(out_fp))

    def scrape_publications(self, pdf_dir, txt_dir):
        """Downloads pdfs for all Technical Reports and scrapes abstracts
        for all other publications listed.

        Parameters
        ----------
        pdf_dir: str
            Directory where the .pdf files should be saved.
        txt_dir: str
            Directory where the .txt files should be saved.
        """

        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)
        url_list = self.all_links[:20]  # quantity control here #

        for pub in url_list:
            with urlopen(pub) as page:
                html = page.read().decode("utf-8")
            pubs_soup = BeautifulSoup(html, "html.parser")

            category = self._scrape_category(pubs_soup)

            if category == 'Technical Report':
                self.download_pdf(pdf_dir, txt_dir, pubs_soup)
            else:
                fn = os.path.basename(pub) + '_abstract.txt'
                self.scrape_abstract(txt_dir, fn, pubs_soup)

        return logger.info('Finished processing publications')


class ResearcherProfiles():
    """
    Class to handle researcher profiles portion of the NREL researcher hub.
    """
    BASE_URL = "https://research-hub.nrel.gov/en/persons/?page=0"

    def __init__(self, url, n_pages=1, txt_dir='./ew_txt'):
        """
        Parameters
        ----------
        url : str
            Research hub profiles URL, most likely
            https://research-hub.nrel.gov/en/persons/
        n_pages : int
            Number of pages to get from the API. Typical response has 50
            entries per page. Default of 1 ensures that this class doesnt hang
            on a million responses.
        txt_dir : str
            File directory where you would like to save output .txt files.
        """

        self.text_dir = txt_dir
        self.profile_links = []
        for p in range(0, n_pages):
            url_base = url + f"?page={p}"
            with urlopen(url_base) as page:
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
        profiles_meta : pd.DataFrame
            Dataframe containing metadata for researcher profiles.
        """
        url_list = self.profile_links
        profiles_meta = pd.DataFrame(columns=('title', 'nrel_id',
                                              'email', 'url', 'fn',
                                              'category'
                                              ))
        for link in url_list[:20]:  # quantity control here #
            with urlopen(link) as page:
                html = page.read().decode("utf-8")
            meta_soup = BeautifulSoup(html, "html.parser")

            title = meta_soup.find('h1').text
            email_target = meta_soup.find('a', {'class': 'email'})
            if email_target:
                email = meta_soup.find('a',
                                       {'class': 'email'}
                                       ).text.replace('nrelgov', '@nrel.gov')
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
        """Scrapes name and position for each researcher.

        Parameters
        ----------
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given researcher.

        Returns
        -------
        intro : str
            String containing researchers name and position.
        """

        r = soup_inst.find('h1').text

        if soup_inst.find('span', {'class': 'job-title'}):
            j = soup_inst.find('span', {'class': 'job-title'}).text
            intro = (f'The following is brief biography for {r} '
                     f'who is a {j} at the National Renewable Energy '
                     f'Laboratory:\n')
        else:
            intro = (f'The following is brief biography for {r}'
                     f'who works for the National Renewable Energy '
                     f'Laboratory:\n')

        return intro

    def _scrape_bio(self, soup_inst):
        """Scrapes 'Personal Profile' section for each researcher.

        Parameters
        ----------
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given researcher.

        Returns
        -------
        bio : str
            String containing background text from profile.
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
        """Scrapes sections such as 'Professional Experience' and
        'Research Interests'

        Parameters
        ----------
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given researcher.
        heading: str
            Section to scrape. Should be 'Professional Experience' or
            'Research Interests'

        Returns
        -------
        text : str
            String containing contents from the experience section.
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
        """Scrapes and reformats 'Education/Academic Qualification'
        section for each researcher.

        Parameters
        ----------
        soup_inst : bs4.BeautifulSoup
            Instantiated beautiful soup instance for the url associated with a
            given researcher.

        Returns
        -------
        full_text : str
            String containing researcher's education (level, focus,
            and institution).
        """
        r = soup_inst.find('h1').text
        target = soup_inst.find('h3',
                                string='Education/Academic Qualification')

        full_text = ''
        if target:
            for sib in target.find_next_siblings():
                t = sib.text
                if len(t.split(',')) >= 3:
                    level = t.split(',')[0]
                    deg = t.split(',')[1]
                    inst = ','.join(t.split(',')[2:])

                    text = (f"{r} received a {level} degree in {deg} "
                            f"from the {inst}. ")
                elif len(t.split(',')) == 2:
                    level = t.split(',')[0]
                    inst = t.split(',')[1]

                    text = f"{r} received a {level} degree from the {inst}. "

                full_text = full_text + text

        return full_text

    def _scrape_publications(self, profile_url):
        """Scrapes the name of each publication that a
        researcher contributed to.

        Parameters
        ----------
        profile_url : str
            Link to a specific researchers profile.

        Returns
        -------
        text : str
            String containing names of all publications for a given researcher.
        """
        pubs_url = profile_url + '/publications/'
        with urlopen(pubs_url) as page:
            html = page.read().decode("utf-8")
        pubs_soup = BeautifulSoup(html, "html.parser")

        r = pubs_soup.find('h1').text
        target = pubs_soup.find_all('h3', {'class': 'title'})

        pubs = []
        if target:
            for p in target:
                pubs.append(p.text)

            pubs = ', '.join(pubs)
            text = (f'{r} has contributed to the following '
                    f'publications: {pubs}.')
        else:
            text = ''

        return text

    def _scrape_similar(self, profile_url):
        """Scrapes the names listed under the 'Similar Profiles' section.

        Parameters
        ----------
        profile_url : str
            Link to a specific researchers profile.

        Returns
        -------
        text : str
            String containing names of similar researchers.
        """
        sim_url = profile_url + '/similar/'
        with urlopen(sim_url) as sim_page:
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
        """Scrapes profiles for each researcher.

        Parameters
        ----------
        out_dir: str
            Directory where the .txt files should be saved.
        """
        os.makedirs(out_dir, exist_ok=True)
        url_list = self.profile_links[:20]  # quantity control here #

        for i, prof in enumerate(url_list):
            f = os.path.basename(prof) + '.txt'
            txt_fp = os.path.join(out_dir, f)
            if not os.path.exists(txt_fp):
                with urlopen(prof) as page:
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

                full_txt = (intro + bio + '\n' + exp + '\n'
                            + interests + '\n' + edu + '\n'
                            + pubs + '\n' + similar)

                with open(txt_fp, "w") as text_file:
                    text_file.write(full_txt)
                logger.info('Profile {}/{}: {} saved to '
                            '{}'.format(i + 1, len(url_list),
                                        r, txt_fp))

            else:
                logger.info('Profile {}/{} already '
                            'exists.'.format(i + 1, len(url_list)))
        return logger.info('Finished processing profiles')


class ProfilesRecord(dict):
    """Class to handle a single profiles as dictionary data"""
    def __init__(self, record):
        """
        Parameters
        ----------
        record : dict
            Profile in dict form, typically a response from the API.
        """
        assert isinstance(record, dict)

        super().__init__(**record)

    @staticmethod
    def clean_text(html_text):
        """Clean html text from API response
        Parameters
        ----------
        html_text : str
            Text containing html characters.
        Returns
        -------
        clean : str
            Text with html characters removed.
        """
        clean = re.sub(r'<.*?>', '', html_text)
        clean = clean.replace('\xa0', ' ')

        return clean

    @property
    def title(self):
        """Get the full name of this researcher.
        Returns
        -------
        full : str
            Full name of researcher.
        """
        names = self.get('name')
        first = names.get('firstName')
        last = names.get('lastName')
        full = first + ' ' + last

        return full

    @property
    def email(self):
        """Get the email address of this researcher.
        Returns
        -------
        email : str
            Email address of researcher.
        """
        orgs = self.get('staffOrganisationAssociations')
        if orgs:
            emails_dict = orgs[0].get('emails')
            if emails_dict:
                email = emails_dict[0].get('value').get('value')

                return email

    @property
    def url(self):
        """Get the url or this researcher's profile.
        Returns
        -------
        url : str
            URL to researcher's profile.
        """
        info = self.get('info')
        url = info.get('portalUrl')

        return url

    @property
    def id(self):
        """Get API ID of researcher.
        Returns
        -------
        id : str
            Researcher ID.
        """
        id = self.get('pureId')

        return id

    @property
    def position(self):
        """Get the position of this researcher.
        Returns
        -------
        position : str
            Researcher's position.
        """
        org = self.get('staffOrganisationAssociations')
        if org:
            info = org[0].get('jobDescription')
            text = info.get('text')[0]
            position = text.get('value')

            return position

    @property
    def profile_information(self):
        """Get key profile information for this record:
        Personal Profile, Research Interests, Professional Experience.
        Returns
        -------
        bio : str
            Researcher's profile text.
        interests : str
            Text from Research Interests section.
        experience : str
            Text from Professional Experience section.
        """
        prof = self.get('profileInformations')

        bio = None
        interests = None
        experience = None

        if prof:
            for section in prof:
                type = section.get('type').get('term')
                if 'Personal Profile' in str(type):
                    info = section.get('value').get('text')[0]
                    bio = info.get('value')
                    bio = self.clean_text(bio)

                if 'Research Interests' in str(type):
                    info = section.get('value').get('text')[0]
                    interests = info.get('value')
                    interests = self.clean_text(interests)

                if 'Professional Experience' in str(type):
                    info = section.get('value').get('text')[0]
                    experience = info.get('value')
                    experience = self.clean_text(experience)
    
        return bio, interests, experience

    @property
    def education(self):
        """Get the education information of this researcher.
        Returns
        -------
        levels : list
            Degree levels, ex: Master, Bachelor, PhD
        degs : list
            Area of study, ex: Mechanical Engineering
        schools : list
            School awarding the degree.
        """
        researcher_name = self.title
        edu = self.get('educations')
        out_strings = []

        if edu:
            for e in edu:
                if e.get('projectTitle'):
                    quali = e.get('qualification')
                    level = quali.get('term').get('text')[0].get('value')
                    deg = e.get('projectTitle').get('text')[0].get('value')
                    org = e.get('organisationalUnits')[0]
                    name = org.get('externalOrganisationalUnit').get('name')
                    school = name.get('text')[0].get('value')

                    deg_string = (f'{researcher_name} has a {level} '
                                  f'degree in {deg} from {school}. ')
                    out_strings.append(deg_string)
                else:
                    quali = e.get('qualification')
                    level = quali.get('term').get('text')[0].get('value')
                    org = e.get('organisationalUnits')[0]
                    name = org.get('externalOrganisationalUnit').get('name')
                    school = name.get('text')[0].get('value')

                    deg_string = (f'{researcher_name} has a {level} '
                                  f'degree from {school}. ')
                    out_strings.append(deg_string)

            return out_strings

    @property
    def publications(self):
        """Get the publications this researcher contributed to.
        Returns
        -------
        pubs : list
            All publications associated with this researcher.
        """
        api_key = os.getenv("RHUB_API_KEY")
        assert api_key is not None, "Please set RHUB_API_KEY!"

        id = self.get('pureId')
        url = (f'https://research-hub.nrel.gov/ws/api/524/persons/'
               f'{id}/research-outputs?size=100'
               f'&apiKey={api_key}')
        session = requests.Session()
        response = session.get(url,  headers={'Accept': 'application/json'})

        content = response.json()['items']

        pubs = []

        for pub in content:
            title = pub.get('title').get('value')
            pubs.append(title)

        return pubs

    def download(self, fp):
        """Download text file containing researchers profile information.
        """
        name = self.title

        if self.position:
            full = (f"The following is a brief biography for {name} "
                    f"who is a {self.position} for the National Renewable "
                    f"Energy Laboratory: ")
        else:
            full = (f"The following is a brief biography for {name} "
                    f"who works for the National Renewable "
                    f"Energy Laboratory: ")

        profile, interests, experience = self.profile_information

        if profile:
            full += profile + ' '

        if interests:
            research = (f"{name}'s research interests include: "
                        f"{interests}. ")
            full += research

        if experience:
            research = (f"{name}'s professional experience includes: "
                        f"{experience}. ")
            full += research

        if self.education:
            for edu in  self.education:
                full += edu

        if self.publications:
            publications = (f"{name} has been involved in the following "
                            f"pubications: {', '.join(self.publications)}. ")
            full += publications

        with open(fp, "w") as text_file:
            text_file.write(full)


class ProfilesList(list):
    """Class to retrieve and handle multiple profiles from an API URL."""
    def __init__(self, url, n_pages=1):
        """
        Parameters
        ----------
        url : str
            Research Hub API URL to request, see this for details:
                https://research-hub.nrel.gov/ws/api/524/api-docs/index.html
        n_pages : int
            Number of pages to get from the API. Typical response has 20
            entries per page. Default of 1 ensures that this class doesnt hang
            on a million responses.
        """

        self.url = url
        self._session = requests.Session()
        self._response = None
        self._n_pages = 0
        self._iter = 0

        records = self._get_first()
        for page in self._get_pages(n_pages=n_pages):
            records += page
        records = [ProfilesRecord(single) for single in records]
        super().__init__(records)

    def _get_first(self):
        """Get the first page of Profiles.

        Returns
        -------
        list
        """
        self._response = self._session.get(self.url,
                                           headers={'Accept': 'application/json'})
        resp = self._response.json()

        if not self._response.ok:
            msg = ('API Request got error {}: "{}"'
                   .format(self._response.status_code,
                           self._response.reason))
            raise RuntimeError(msg)
        first_page = self._response.json()['items']

        self._n_pages = 1

        if not 'last' in self._response.links:
            count_pages = resp['count']/resp['pageInformation'].get('size')
            self._n_pages = math.ceil(count_pages)
        else:
            url = self._response.links['last']['url']
            self._n_pages = int(url.split('page=')[-1])

        logger.debug('Found approximately {} records.'
                     .format(self._n_pages * len(first_page)))

        return first_page

    def _get_pages(self, n_pages):
        """Get response pages up to n_pages from Research Hub.

        Parameters
        ----------
        n_pages : int
            Number of pages to retrieve

        Returns
        -------
        next_pages : list
            This function will return a generator of next pages, each of which
            is a list of profiles.
        """
        if n_pages > 1:
            for page in range(2, self._n_pages + 1):
                if page <= n_pages:
                    next_page = self._session.get(self.url,
                                                  params={'page': page},
                                                 headers={'Accept': 'application/json'})
                    next_page = next_page.json()['items']
                    yield next_page
                else:
                    break

    def meta(self):
        """Get a meta dataframe with details on all of the profiles.

        Returns
        -------
        df : pd.DataFrame
            Dataframe containing all metadata information.
        """
        i = 0
        attrs = ('title', 'email', 'url', 'id')
        df = pd.DataFrame(columns=attrs)
        for record in self:
            for attr in attrs:
                out = getattr(record, attr)
                if not isinstance(out, str):
                    out = json.dumps(out)
                df.at[i, attr] = out
            df.at[i, 'fn'] = f'{record.id}.txt'
            df.at[i, 'category'] = 'Researcher Profile'
            i += 1

        return df

    def download(self, out_dir):
        """Download all profiles from the records in this object into a
        directory. TXT files will be given file names based on researcher ID.

        Parameters
        ----------
        out_dir : str
            Directory to download TXT files to. This directory will be created
            if it does not already exist.
        """
        os.makedirs(out_dir, exist_ok=True)
        for record in self:
            fn = record.title.lower().replace(' ', '-')
            fp_out = os.path.join(out_dir, fn + '.txt')
            if not os.path.exists(fp_out):
                try:
                    record.download(fp_out)
                except Exception as e:
                    print(f"Could not download {record.title} with error {e}")
                    logger.exception('Could not download profile ID {}: {}'
                                     .format(record.title, e))
        logger.info('Finished Profiles download!')


class PublicationsRecord(dict):
    """Class to handle a single publication as dictionary data"""
    def __init__(self, record):
        """
        Parameters
        ----------
        record : dict
            Research Hub record in dict form, typically a response from API.
        """
        assert isinstance(record, dict)
        super().__init__(**record)

    @property
    def title(self):
        """Get the title of this publication.
        Returns
        -------
        title : str
            Publication title.
        """
        title = self.get('title').get('value')

        return title

    @property
    def year(self):
        """Get the publish year.
        Returns
        -------
        year : int
            Year of publication.
        """
        status = self.get('publicationStatuses')[0]
        year = status.get('publicationDate').get('year')

        return year

    @property
    def url(self):
        """Get the url associated with the publication.
        Returns
        -------
        url : str
            Publication URL.
        """
        info = self.get('info')
        url = info.get('portalUrl')

        return url

    @property
    def id(self):
        """Get the 'NREL Publication Number' for
        this record.
        Returns
        -------
        id : str
            Publication Number.
        """
        group = self.get('keywordGroups')[0]
        cont = group.get('keywordContainers')[0]
        id = cont.get('freeKeywords')[0].get('freeKeywords')[0]
        id = id.replace('/', '-')

        return id

    @property
    def authors(self):
        """Get the names of all authors for a publication.
        Returns
        -------
        out : str
            String containing author names.
        """
        pa = self.get('personAssociations')

        authors = []

        for r in pa:
            r.get('name')
            first = r.get('name').get('firstName')
            last = r.get('name').get('lastName')

            full = first + ' ' + last

            authors.append(full)

            out = ', '.join(authors)

        return out

    @property
    def category(self):
        """Get the category for this publication.
        Returns
        -------
        cat : str
            Publication category, ex: Technical Report, Article.
        """
        type = self.get('type')
        term = type.get('term')
        cat = term.get('text')[0].get('value')

        return cat

    @property
    def links(self):
        """Get the doi and pdf links for a publication.
        Returns
        -------
        doi : str
            doi link for publication.
        pdf_url : str
            pdf link for publication.
        """
        ev = self.get('electronicVersions')

        doi = None
        pdf_url = None

        for l in ev:
            if l.get('doi'):
                doi = l.get('doi')
            if l.get('link'):
                pdf_url = l.get('link')

        return doi, pdf_url

    @property
    def abstract(self):
        """Get the abstract text for this publication.
        Returns
        -------
        value : str
            String containing abstract text.
        """
        abstract = self.get('abstract')
        text = abstract.get('text')[0]
        value = text.get('value')

        return value

    def save_abstract(self, abstract_text, out_fp):
        """Download abstract text to .txt file to the directory
        provided.
        Parameters
        ----------
        abstract_text : str
            String with abstract text.
        out_dir : str
            Directory to download TXT files to.
        """
        title = self.title
        full = f"The report titled {title} can be summarized as follows: "
        full += abstract_text

        with open(out_fp, "w") as text_file:
            text_file.write(full)


    def download(self, pdf_dir, txt_dir):
        """Download PDFs and TXT files to the directories provided. If a record
        does not fit the criteria for PDF download,a TXT file with the record
        abstract will be saved to the TXT directory.
        Parameters
        ----------
        pdf_dir : str
            Directory for pdf download.
        txt_dir : str
            Directory for txt download.
        """

        category = self.category
        title = self.title
        pdf_url = self.links[1]
        abstract = self.abstract

        if category != 'Technical Report':
            fn = self.id.replace('/', '-') + '.txt'
            fp = os.path.join(txt_dir, fn)
            if not os.path.exists(fp):
                if abstract:
                    self.save_abstract(abstract, fp)
        else:
            if pdf_url and pdf_url.endswith('.pdf'):
                fn = self.id.replace('/', '-') + '.pdf'
                fp = os.path.join(pdf_dir, fn)
                if not os.path.exists(fp):
                    session = requests.Session()
                    response = session.get(pdf_url)
                    with open(fp, 'wb') as f_pdf:
                        f_pdf.write(response.content)
            else:
                fn = title.lower().replace(' ', '-') + '.txt'
                fp = os.path.join(pdf_dir, fn)
                self.save_abstract(abstract, fp)

class PublicationsList(list):
    """Class to retrieve and handle multiple publications from an API URL."""
    def __init__(self, url, n_pages=1):
        """
        Parameters
        ----------
        url : str
            Research Hub API URL to request, see this for details:
                https://research-hub.nrel.gov/ws/api/524/api-docs/index.html
        n_pages : int
            Number of pages to get from the API. Typical response has 20
            entries per page. Default of 1 ensures that this class doesnt hang
            on a million responses.
        """

        self.url = url
        self._session = requests.Session()
        self._response = None
        self._n_pages = 0
        self._iter = 0

        records = self._get_first()
        for page in self._get_pages(n_pages=n_pages):
            records += page
        records = [PublicationsRecord(single) for single in records]
        super().__init__(records)

    def _get_first(self):
        """Get the first page of publications

        Returns
        -------
        first_page : list
            Publication records as list.
        """
        self._response = self._session.get(self.url,
                                           headers={'Accept': 'application/json'})

        resp = self._response.json()

        if not self._response.ok:
            msg = ('API Request got error {}: "{}"'
                   .format(self._response.status_code,
                           self._response.reason))
            raise RuntimeError(msg)
        first_page = self._response.json()['items']

        self._n_pages = 1

        if not 'last' in self._response.links:
            count_pages = resp['count']/resp['pageInformation'].get('size')
            self._n_pages = math.ceil(count_pages)
        else:
            url = self._response.links['last']['url']
            self._n_pages = int(url.split('page=')[-1])

        logger.debug('Found approximately {} records.'
                     .format(self._n_pages * len(first_page)))

        return first_page

    def _get_pages(self, n_pages):
        """Get response pages up to n_pages from Research Hub.

        Parameters
        ----------
        n_pages : int
            Number of pages to retrieve

        Returns
        -------
        next_pages : list
            This function will return a generator of next pages, each of which
            is a list of records.
        """
        if n_pages > 1:
            for page in range(2, self._n_pages + 1):
                if page <= n_pages:
                    next_page = self._session.get(self.url,
                                                  params={'page': page},
                                                 headers={'Accept': 'application/json'})

                    next_page = next_page.json()['items']
                    yield next_page
                else:
                    break

    def meta(self):
        """Get a meta dataframe with details on all of the publications.

        Returns
        -------
        df : pd.DataFrame
            Dataframe containing all metadata information.
        """
        i = 0
        attrs = ('title', 'year', 'url', 'id', 'category', 'authors')
        df = pd.DataFrame(columns=attrs)
        for record in self:
            doi = record.links[0]
            pdf_url = record.links[1]
            for attr in attrs:
                out = getattr(record, attr)
                if not isinstance(out, str):
                    out = json.dumps(out)
                df.at[i, attr] = out
            df.at[i, 'doi'] = doi
            df.at[i, 'pdf_url'] = pdf_url
            i += 1

        return df

    def download(self, pdf_dir, txt_dir):
        """Download all PDFs and abstract TXTs from the records in this object
        into a directory. Files will be given file names based on their record ID.

        Parameters
        ----------
        pdf_dir : str
            Directory to download PDFs to. This directory will be created if it
            does not already exist.
        txt_dir : str
            Directory to download TXTs to. This directory will be created if it
            does not already exist.
        """
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)
        for record in self:
            try:
                record.download(pdf_dir, txt_dir)
            except Exception as e:
                print(f"Could not download {record.title} with error {e}")
                logger.exception('Could not download {}: {}'
                                    .format(record.title, e))
        logger.info('Finished publications download!')
