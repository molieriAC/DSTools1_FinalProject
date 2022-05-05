import pandas as pd
import sqlite3
import requests
from bs4 import BeautifulSoup
from config import WebScrapingAPIkey

def web_scrape_api_call(url_to_scrape):
    '''
    sends the url that we would like to scrape to the webscrapingapi
    so that our calls can be ananomyzed. 
    '''
    url = "https://api.webscrapingapi.com/v1"
    params = {
    "api_key":WebScrapingAPIkey,
    "url":url_to_scrape
    }
    response = requests.request("GET", url, params=params)
    return response

# Get company names from sqlite database
def pull_company_names(loc):
    """function to pull company names from a the sqlite databse that contains job openings data

    Args:
        loc (string): file path for the sqlite database
        
    Return:
        DF of company names from the database
    """
    # set up the sqlite connection
    cnct = sqlite3.connect(loc)
    # use row_factory so the response isn't a list of tuples
    cnct.row_factory = lambda cursor, row: row[0]
    cursor = cnct.cursor()
    
    # create the query, keeping only unique company names
    cursor.execute('SELECT DISTINCT company_name FROM indeed_jobs')
    comps = list(cursor.fetchall())

    # there may be characters that will cause the search to fail fix this by building a dataframe that has the original name and stripped name
    compsDf = pd.DataFrame(comps, columns = ['companies'])
    compsDf['nameFix'] = compsDf['companies'].str.replace("[,./\()'\"-]", ' ', regex=True).str.lower()
    
    # # output dataframe to file for review
    compsDf.to_csv('checks/companies.txt')
        
    return compsDf

# search glassdoor for company IDs
def company_URL_search(nameList: pd.DataFrame):
    """glassdoor requires their unique ID and official name to be present in a URL for a company scraping the unique URL from search will allow us to pull specific company information later on.

    Args:
        nameList (dataframe): contains list of company names that we are finding the URLs for
        
    Return:
        an updated version of the original dataframe that contains the partial urls
    """

    nameListNew = nameList.copy()
    nameListNew['linkPart'] = ""
    counter = 0
    for name in nameListNew['nameFix'].to_list():
        counter += 1
        # html = requests.get(f"https://www.glassdoor.com/Search/results.htm?keyword={name}", headers = {'User-agent': 'Mozilla/5.0'})
        
        # # try using webscraping api
        # # html = web_scrape_api_call(f"https://www.google.com/search?q=glassdoor working at {name}")
        
        # nameSoup = BeautifulSoup(html.content, 'html.parser')
        # # loop through the returned html and save the proper links
        # for a in nameSoup.find_all('a', href=True):
        #     if "/Overview/" in a['href']:
        #         nameListNew.loc[nameListNew['nameFix'] == name, 'linkPart'] = a['href']
        #         # for now we're only going to take the top result from search
        # time.sleep(10)
        
        # do a google search to compare?:
        # html2 = requests.get(f"https://www.google.com/search?q=glassdoor working at {name}", headers = {'User-agent': 'Mozilla/5.0'})
        html2 = web_scrape_api_call(f"https://www.google.com/search?q=glassdoor working at {name}")
        
        googSoup = BeautifulSoup(html2.content, 'html.parser')
        # loop through the returned html and save the proper links
        for a in googSoup.find_all('a', href=True):
            if "/Overview/" in a['href']:
                workAtIndex = a['href'].index('Working-at-')
                htmIndex = a['href'].index('.htm')
                trimLink = a['href'][workAtIndex+11:htmIndex+4]
                print(counter, trimLink)
                nameListNew.loc[nameListNew['nameFix'] == name, 'linkPart'] = trimLink
                # for now we're only going to take the top result from search
                break
            
    # save link parts to file for review
    nameListNew.to_csv('checks/scrapeTest.csv')     
    
    return nameListNew

# pull company benefits from glassdoor
def company_benefits_scraper(URLList):
    """takes a list of company page URL from glassdoor and returns a dataframe of their benefits ratings

    Args:
        URLList (dataframe): contains a variable with company page urls from glassdoor

    Returns:
        dataframe: dataframe of benefits ratings
    """
    # create an empty dataframe for saving
    beneDF = pd.DataFrame(columns=['type', 'rating', 'count_of_ratings', 'linkPart'])
    # Go to each company page and pull the relevant data
    for u in URLList:
        if not u:
            pass
        else:
            # the urls from the scraper are for the company pages.
            print(f"scraping: {u}")
            # html = requests.get(f"https://www.glassdoor.com/Benefits/{u}", headers = {'User-agent': 'Mozilla/5.0'})
            html = web_scrape_api_call(f"https://www.glassdoor.com/Benefits/{u}")

            benefitSoup = BeautifulSoup(html.content, 'html.parser')
            
            # get benefit type data
            benefitTypes = benefitSoup.find_all('div', {"class": 'p-std css-14xbtcm ejjgytf0'})
            
            # save to text for testing
            # with open('checks/check_benefitTypes.txt', "w", encoding="utf-8") as writer:
            #     writer.write(str(benefitTypes))   
            
            #Loop through the benefits data to get benefit type, subtypes, ratings values and counts of ratings (per subtype)
            beneDict = {}
            # benefits have 6 headings loop through them
            for ben in benefitTypes:
                benKey = ben.find('h3', {"data-test": 'benefitsTabNameHeader'}).text
                subTypes = []
                ratingValue = []
                ratingCount = []
                # there can be many different subtypes depending on the company
                for link in ben.find_all('a', href=True):
                    subTypes.append(link.text)
                # get review values per sub-type
                for v in ben.find_all('span', {'data-test' : 'ratingValue'}):
                    ratingValue.append(v.text)
                # get the number of ratings per sub-type for weighted averaging
                for c in ben.find_all('div', {'data-test' : 'ratingCount'}):
                    ratingCount.append(c.text)
                # combine the gathered data into a dictionary to be converted to a dataframe
                subComb = list(map(list, zip(subTypes, ratingValue, ratingCount)))
                for i in subComb:
                    beneDict[f'{benKey} - {i.pop(0)}'] = i

            # add it all to a dataframe
            tempDF = pd.DataFrame.from_dict(beneDict, columns=['rating', 'count_of_ratings'], orient='index')
            tempDF['linkPart'] = u
            tempDF = tempDF.reset_index().rename(columns = {'index':'type'})
            beneDF = pd.concat([beneDF, tempDF], ignore_index=True)
            beneDF['count_of_ratings'] = beneDF['count_of_ratings'].str.replace(" Rating","").str.replace("s","")

    return beneDF

# runs everything
def main():
    # location of the jobs database
    db_loc = 'Data/jobs'
    
    # get company names from the indeed table in the database
    cmp = pull_company_names(db_loc)
    print(f"{len(cmp)} unique companies present in the dataset")
    
    # make a shorter list of companies for testing
    # cmp = cmp.head(20)
    
    # run the company ID search
    compURLs = company_URL_search(cmp)

    # Scrape the ratings
    benefitScrape = company_benefits_scraper(compURLs['linkPart'].to_list())
    
    # Combine the scraped ratings with the original name information (to allow for matching back to the db)
    comb = benefitScrape.merge(compURLs, on='linkPart')
    
    # export to csv
    comb.to_csv('checks/benes.csv', index_label="TblID")
    
    #Add ratings to a new table in the database
    cnct = sqlite3.connect(db_loc)
    comb.to_sql('ratings', cnct, if_exists='replace', index_label="TblID")

if __name__ == "__main__":
    main()
    