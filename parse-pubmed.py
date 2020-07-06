from bs4 import BeautifulSoup, NavigableString
import sqlite3

xmlfile = 'pubmed_result.xml'
dbfile = 'pubmed.db'
maxparse = 1000000
parsestep = 1000
startpoint = 0
numlines = 0
badarticles = 0
goodarticles = 0

# prepopulate in-memory authors, mesh terms, journals, articles
conn = sqlite3.connect(dbfile)
c = conn.cursor()
authorDict = {}
c.execute('SELECT id, name FROM Authors')
for row in c:
    authorDict[row[1].strip()] = int(row[0])

meshtermDict = {}
c.execute('SELECT id, headingqualifier FROM MeSHTerm')
for row in c:
    meshtermDict[row[1].strip()] = int(row[0])

meshHeadingDict = {}
c.execute('SELECT id, heading FROM MeSHHeading')
for row in c:
    meshHeadingDict[row[1].strip()] = int(row[0])

journalDict = {}
c.execute('SELECT id, name FROM Journals')
for row in c:
    journalDict[row[1].strip()] = int(row[0])

articleDict = {}
c.execute('SELECT id, pubmedid FROM Articles')
for row in c:
    articleDict[row[1].strip()] = int(row[0])

with open(xmlfile, 'r', encoding='utf-8', errors='surrogateescape') as f:
    l = f.readline()
    article = ''
    articles = 1
    while l:
        article = article + l
        if '</PubmedArticle>' in l: 
            # progress
            if articles % parsestep == 0:
                print('Article:', articles, 'Good Articles:', goodarticles, 'Bad Articles:', badarticles)

            if articles >= maxparse:
                break

            if articles >= startpoint:
                soup = BeautifulSoup(' '.join(article.split()).replace('> <','><'), 'xml')

                # extract doi and pubmedid
                doi = ''
                pubmedid = ''
                for tag in soup.find_all('ArticleId'):
                    if 'IdType' in tag.attrs:
                        if tag['IdType'] == 'doi':
                            doi = tag.get_text().strip()
                        if tag['IdType'] == 'pubmed':
                            pubmedid = tag.get_text().strip()

                # extract title
                title = ''
                titletag = soup.find('ArticleTitle')
                title = titletag.get_text() if titletag else None

                # extract date
                year = 0
                month = 0
                day = 0
                datetag = soup.find('DateCompleted')
                if datetag:
                    year = int(datetag.find('Year').get_text())
                    month = int(datetag.find('Month').get_text())
                    day = int(datetag.find('Day').get_text())

                # extract abstract
                abstracttag = soup.find('Abstract')
                abstract = ''
                if not abstracttag:
                    for tag in soup.find_all('OtherAbstract'):
                        if 'Language' in tag.attrs and tag['Language'] == 'eng':
                            abstracttag = tag
                            break
                
                if abstracttag:
                    if len(abstracttag.contents) == 1:
                        abstract = abstracttag.get_text()
                    else:
                        for child in abstracttag.contents:
                            if child.name == 'AbstractText':
                                if 'Label' in child.attrs:
                                    if len(abstract) > 1:
                                        abstract = abstract + ' ' + child['Label'] + ': ' + child.get_text()
                                    else:
                                        abstract = abstract + child['Label'] + ': ' + child.get_text()
                                else:
                                    abstract = abstract + child.get_text()
                            elif isinstance(child, NavigableString):
                                abstract = abstract + child.string if child.string else ''

                # extract journal
                journalname = ''
                journalnlmid = ''
                journaltag = soup.find('MedlineJournalInfo')
                if journaltag:
                    journalname = journaltag.find('MedlineTA').get_text().strip()
                    journalnlmid = journaltag.find('NlmUniqueID').get_text().strip()

                # extract authors
                authors = []
                authortag = soup.find('AuthorList')
                if authortag:
                    for child in authortag.find_all('Author'):
                        name = ''
                        if child.find('ForeName'):
                            name = name + child.find('ForeName').string
                        
                        if child.find('LastName'):
                            name = name + ' ' + child.find('LastName').string

                        if child.find('CollectiveName'):
                            if len(name) > 0:
                                name = name + ' ' + child.find('CollectiveName').string
                            else:
                                name = child.find('CollectiveName').string
                        
                        if name != '':
                            authors.append(name)

                # extract MeSH terms
                meshterms = []
                meshheadings = []
                meshqualifiers = []
                meshtag = soup.find('MeshHeadingList')
                if meshtag:
                    for child in meshtag.find_all('MeshHeading'):
                        descriptor = child.find('DescriptorName')
                        if descriptor:
                            qualifiers = child.find_all('QualifierName')
                            if len(qualifiers) == 0:
                                meshterms.append(descriptor.get_text())
                                meshheadings.append(descriptor.get_text())
                                meshqualifiers.append('')
                            elif len(qualifiers) == 1:
                                meshterms.append(descriptor.get_text() + ' - ' + qualifiers[0].get_text())
                                meshheadings.append(descriptor.get_text())
                                meshqualifiers.append(qualifiers[0].get_text())
                            else:
                                for qualifier in qualifiers:
                                    meshterms.append(descriptor.get_text() + ' - ' + qualifier.get_text())
                                    meshheadings.append(descriptor.get_text())
                                    meshqualifiers.append(qualifier.get_text())

                # check if good
                if pubmedid != '' and pubmedid not in articleDict and len(abstract) > 50 and len(authors) >= 1 and \
                    len(meshterms) >= 1 and len(title) > 3 and journalname != '' and year != 0 and month != 0 and day != 0:

                    # get journal ID & enter journal if not already in
                    if journalname not in journalDict:
                        c.execute("INSERT INTO Journals (name, nlmid) VALUES (?, ?)", (journalname, journalnlmid))
                        journalid = c.lastrowid
                        journalDict[journalname] = int(journalid)
                    else:
                        journalid = journalDict[journalname]

                    # enter article
                    c.execute("INSERT INTO Articles (doi, pubmedid, title, abstract, journal, year, month, day) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                        (doi, pubmedid, title, abstract, journalid, year, month, day))
                    articleid = c.lastrowid
                    articleDict[pubmedid] = int(articleid)

                    # go through authors
                    for author in authors:
                        if len(author) > 1:
                            if author not in authorDict:
                                c.execute("INSERT INTO Authors (name) VALUES (?)", (author,))
                                authorid = c.lastrowid
                                authorDict[author] = int(authorid)
                            else:
                                authorid = authorDict[author]
                            
                            c.execute("INSERT INTO ArticleAuthor (articleid, authorid) VALUES (?, ?)", (articleid, authorid))

                    # go through mesh terms:
                    for term, heading, qualifier in zip(meshterms, meshheadings, meshqualifiers):
                        if heading not in meshHeadingDict:
                            c.execute("INSERT INTO MeSHHeading (heading) VALUES (?)", (heading,))
                            meshheadingid = c.lastrowid
                            meshHeadingDict[heading] = c.lastrowid
                        else:
                            meshheadingid = meshHeadingDict[heading]

                        if term not in meshtermDict:
                            c.execute("INSERT INTO MeSHTerm (heading, qualifier, headingqualifier) VALUES (?, ?, ?)", (heading, qualifier, term))
                            meshtermid = c.lastrowid
                            meshtermDict[term] = int(meshtermid)
                        else: 
                            meshtermid = meshtermDict[term]
                        
                        c.execute("INSERT INTO ArticleMeSHTerm (articleid, meshtermid) VALUES (?, ?)", (articleid, meshtermid))
                        c.execute("INSERT INTO ArticleMeSHHeading (articleid, meshheadingid) VALUES (?, ?)", (articleid, meshtermid))
                    
                    conn.commit()
                    goodarticles += 1
                
                else:
                    badarticles += 1

            article = ''
            articles += 1

        l = f.readline()

print('articles:', articles, 'good articles:', goodarticles, 'bad articles:', badarticles)