"""
Code to build Corpus from the researcher hub.
"""
import os
import os.path
import logging
import json
import math
import re
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class ProfilesRecord(dict):
    """Class to handle a single profiles as dictionary data.
    This class requires setting an 'RHUB_API_KEY' environment
    variable to access the Pure Web Service. The API key can be
    obtained by contacting an NREL library representative:
    Library@nrel.gov.
    """
    def __init__(self, record):
        """
        Parameters
        ----------
        record : dict
            Profile in dict form, typically a response from the API.
        """
        api_key = os.getenv("RHUB_API_KEY")
        assert api_key is not None, "Please set RHUB_API_KEY!"
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
    def first_name(self):
        """Get the first name of this researcher.

        Returns
        -------
        first : str
            Full name of researcher.
        """
        names = self.get('name')
        first = names.get('firstName')

        return first

    @property
    def last_name(self):
        """Get the last name of this researcher.

        Returns
        -------
        last : str
            Last name of researcher.
        """
        names = self.get('name')
        last = names.get('lastName')

        return last

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
        email = None
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
        level = self.get('ids')[0]
        id = level.get('value').get('value')

        return id

    @property
    def position(self):
        """Get the position of this researcher.

        Returns
        -------
        position : str
            Researcher's position.
        """
        position = None
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
                try:
                    if e.get('projectTitle'):
                        quali = e.get('qualification')
                        level = quali.get('term').get('text')[0].get('value')
                        deg = e.get('projectTitle').get('text')[0].get('value')
                        org = e.get('organisationalUnits')

                        if org:
                            value = org[0].get('externalOrganisationalUnit')
                            name = value.get('name')
                            school = name.get('text')[0].get('value')
                        else:
                            deg_school = deg
                            deg = deg_school.split(',')[0]
                            school = deg_school.split(',')[1]

                        deg_string = (f'{researcher_name} has a {level} '
                                      f'degree in {deg} from {school}. ')
                        out_strings.append(deg_string)
                    else:
                        quali = e.get('qualification')
                        level = quali.get('term').get('text')[0].get('value')
                        org = e.get('organisationalUnits')[0]
                        org_unit = org.get('externalOrganisationalUnit')
                        name = org_unit.get('name')
                        school = name.get('text')[0].get('value')

                        deg_string = (f'{researcher_name} has a {level} '
                                      f'degree from {school}. ')
                        out_strings.append(deg_string)
                except Exception:
                    pass

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
        response = session.get(url, headers={'Accept': 'application/json'})

        content = response.json()['items']

        pubs = []

        for pub in content:
            title = pub.get('title').get('value')
            pubs.append(title)

        return pubs

    def download(self, fp):
        """Download text file containing researchers profile information.

        Parameters
        ----------
        fp : str
            Filepath to download this record to.
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
            for edu in self.education:
                full += edu

        if self.publications:
            publications = (f"{name} has been involved in the following "
                            f"publications: {', '.join(self.publications)}. ")
            full += publications

        with open(fp, "w") as text_file:
            text_file.write(full)


class ProfilesList(list):
    """Class to retrieve and handle multiple profiles from an API URL.
    This class requires setting an 'RHUB_API_KEY' environment
    variable to access the Pure Web Service. The API key can be
    obtained by contacting an NREL library representative:
    Library@nrel.gov.
    """
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
        api_key = os.getenv("RHUB_API_KEY")
        assert api_key is not None, "Please set RHUB_API_KEY!"

        self.url = url
        self._session = requests.Session()
        self._response = None
        self._n_pages = 0
        self._iter = 0

        records = self._get_all(n_pages)
        records = [ProfilesRecord(single) for single in records]
        records = [prof for prof in records if prof.last_name != 'NREL']
        super().__init__(records)

    def _get_first(self):
        """Get the first page of Profiles.

        Returns
        -------
        first_page : list
            First page of records as a list.
        """
        self._response = self._session.get(
            self.url,
            headers={'Accept': 'application/json'}
        )

        resp = self._response.json()

        if not self._response.ok:
            msg = ('API Request got error {}: "{}"'
                   .format(self._response.status_code,
                           self._response.reason))
            raise RuntimeError(msg)
        first_page = self._response.json()['items']

        self._n_pages = 1

        if 'last' not in self._response.links:
            count_pages = resp['count'] / resp['pageInformation'].get('size')
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
                    next_page = self._session.get(
                        self.url,
                        params={'page': page},
                        headers={'Accept': 'application/json'})

                    next_page = next_page.json()['items']
                    yield next_page
                else:
                    break

    def _get_all(self, n_pages):
        """Get all pages of profiles up to n_pages.

        Parameters
        ----------
        n_pages : int
            Number of pages to retrieve

        Returns
        -------
        all_records : list
            List of all publication records.
        """
        first_page = self._get_first()
        records = first_page

        for page in self._get_pages(n_pages):
            records.extend(page)

        return records

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
            fn = record.id
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
    """Class to handle a single publication as dictionary data.
    This class requires setting an 'RHUB_API_KEY' environment
    variable to access the Pure Web Service. The API key can be
    obtained by contacting an NREL library representative:
    Library@nrel.gov.
    """
    def __init__(self, record):
        """
        Parameters
        ----------
        record : dict
            Research Hub record in dict form, typically a response from API.
        """
        api_key = os.getenv("RHUB_API_KEY")
        assert api_key is not None, "Please set RHUB_API_KEY!"

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
        try:
            group = self.get('keywordGroups')[0]
            cont = group.get('keywordContainers')[0]
            id = cont.get('freeKeywords')[0].get('freeKeywords')[0]
            id = id.replace('/', '-')
        except TypeError:
            id = self.get('externalId')

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

        if not pa:
            return None

        authors = []

        for r in pa:
            name = r.get('name')

            if not name:
                continue

            first = name.get('firstName')
            last = name.get('lastName')
            full = " ".join(filter(bool, [first, last]))

            if not full:
                continue

            authors.append(full)

        out = ', '.join(authors)

        return out

    @property
    def category(self):
        """Get the publication category for this record.

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
        if ev:
            for link in ev:
                if link.get('doi'):
                    doi = link.get('doi')
                if link.get('link'):
                    pdf_url = link.get('link')

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

        if not abstract:
            return None

        text = abstract.get('text')

        if not text:
            return None

        value = text[0].get('value')

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
        does not fit the criteria for PDF download, a TXT file with the record
        abstract will be saved to the TXT directory.

        Parameters
        ----------
        pdf_dir : str
            Directory for pdf download.
        txt_dir : str
            Directory for txt download.
        """

        category = self.category
        pdf_url = self.links[1]
        abstract = self.abstract

        pdf_categories = ['Technical Report', 'Paper', 'Fact Sheet']

        if category not in pdf_categories:
            fn = self.id.replace('/', '-') + '.txt'
            fp = os.path.join(txt_dir, fn)
            if not os.path.exists(fp):
                if abstract:
                    self.save_abstract(abstract, fp)
                else:
                    logger.info(f'{self.title}: does not have an '
                                'abstract to downlod')
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
                fn = self.id.replace('/', '-') + '.txt'
                fp = os.path.join(txt_dir, fn)
                self.save_abstract(abstract, fp)


class PublicationsList(list):
    """Class to retrieve and handle multiple publications from an API URL.
    This class requires setting an 'RHUB_API_KEY' environment
    variable to access the Pure Web Service. The API key can be
    obtained by contacting an NREL library representative:
    Library@nrel.gov.
    """
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
        api_key = os.getenv("RHUB_API_KEY")
        assert api_key is not None, "Please set RHUB_API_KEY!"

        self.url = url
        self._session = requests.Session()
        self._response = None
        self._n_pages = 0
        self._iter = 0

        records = self._get_all(n_pages)
        records = [PublicationsRecord(single) for single in records]
        super().__init__(records)

    def _get_first(self):
        """Get the first page of publications

        Returns
        -------
        first_page : list
            Publication records as list.
        """
        self._response = self._session.get(
            self.url,
            headers={'Accept': 'application/json'})

        resp = self._response.json()

        if not self._response.ok:
            msg = ('API Request got error {}: "{}"'
                   .format(self._response.status_code,
                           self._response.reason))
            raise RuntimeError(msg)
        first_page = self._response.json()['items']

        self._n_pages = 1

        if 'last' not in self._response.links:
            count_pages = resp['count'] / resp['pageInformation'].get('size')
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
                    next_page = self._session.get(
                        self.url,
                        params={'page': page},
                        headers={'Accept': 'application/json'})

                    next_page = next_page.json()['items']
                    yield next_page
                else:
                    break

    def _get_all(self, n_pages):
        """Get all pages of publications up to n_pages.

        Parameters
        ----------
        n_pages : int
            Number of pages to retrieve

        Returns
        -------
        all_records : list
            List of all publication records.
        """
        first_page = self._get_first()
        records = first_page

        for page in self._get_pages(n_pages):
            records.extend(page)

        return records

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
        """Download all PDFs and abstract TXTs from the records in this
        objectbinto a directory. Files will be given file names based
        on their record ID.

        Parameters
        ----------
        pdf_dir : str
            Directory to download PDFs to. This directory will be created
            if it does not already exist.
        txt_dir : str
            Directory to download TXTs to. This directory will be created
            if it does not already exist.
        """
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)
        for record in self:
            try:
                record.download(pdf_dir, txt_dir)
            except Exception as e:
                logger.exception('Could not download {}: {}'
                                 .format(record.title, e))
        logger.info('Finished publications download!')
