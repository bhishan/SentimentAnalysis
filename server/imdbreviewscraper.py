from bs4 import BeautifulSoup
import re
import urllib2
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

csvwriter = csv.writer(file('scraped.csv', 'wb'))
#csvwriter = csv.writer(file('scraped.tsv', 'wb'), delimiter='\t')
headerrows = ['id', 'review']
csvwriter.writerow(headerrows)

def scrape_individual_review_page(url):
    resp_query = urllib2.urlopen(url)
    resp = resp_query.read()
    soup = BeautifulSoup(resp)
    required_content = soup.find('div', attrs={'id':'tn15content'})
    for each_review in required_content.findAll('p'):
        required_text = each_review.text
        required_text.decode('utf-8', 'ignore')
        #print required_text
        if "review may contain spoilers" in required_text or "Add another review" in required_text:
            print "invalid review"
        else:
            eachrow = ["1", required_text]
            csvwriter.writerow(eachrow)

def scrape_reviews(url):
    resp_query = urllib2.urlopen(url)
    resp = resp_query.read()
    soup = BeautifulSoup(resp)
    required_content = soup.find('div', attrs={'id':'tn15content'})
    all_tables = required_content.findAll('table')
    required_table = all_tables[-2]
    total_pages = required_table.find('font').text
    total_pages = re.sub("[^0-9]", "", total_pages)
    total_pages = total_pages[1:]
    total_pages = int(total_pages)
    processed_url = (url.split('?'))[0]
    #for i in range(total_pages):
    for i in range(2):

        new_url = processed_url + '?start=' + str(i*10)
        scrape_individual_review_page(new_url)
    



#scrape_reviews("http://www.imdb.com/title/tt0232500/reviews?ref_=tt_urv")
