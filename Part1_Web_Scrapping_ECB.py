from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import requests
from bs4 import BeautifulSoup
headers = {'User-Agent': 'Chrome'}

import pandas as pd
import logging
from selenium.webdriver.remote.remote_connection import LOGGER
LOGGER.setLevel(logging.WARNING)

from bs4 import BeautifulSoup

driver = webdriver.Firefox()
driver.get("https://www.ecb.europa.eu/press/pressconf/html/index.en.html")

import time
for x in range(0, 10000, 200):
    driver.execute_script("window.scrollBy(0, " +str(x)+");")
    time.sleep(1)

html = driver.page_source
title = driver.title
soup = BeautifulSoup(html)

v = soup.prettify()

url_list = []
#date_list = []

for y in soup.find_all("div", {"class" : "title"}):
    for links in y.find_all('a', href=True):
        full_url = driver.current_url[:driver.current_url.find('/press')+1] + links['href'][1:]
        #url_list.append(full_url)
        if "monetary-policy-statement" in full_url:
               if full_url.endswith(".en.html"):
                   url_list.append(full_url)

####################### loop over all urls to scrap info

rows = []

for url in url_list:
    driver.get(url)
    r = BeautifulSoup(driver.page_source, 'html.parser')

    date_start = url.find("is") + 2  if "is" in url else url.find("sp") + 2
    date = url[date_start:date_start + 6]
 
    title = ""
    content = []
    
    for x in r.find_all("main"):
        for y in x.find_all("div", {"class" : "title"}):
            for z in y.find_all("h1"):
                title = z.text
        
    for x in r.find_all("main"):
        for y in x.find_all("p"):
            if "ecb-publicationDate" not in y.get("class", []) and "disclaimer center" not in y.get("class", []):
            #if "ecb-publicationDate" not in y.get("class", []):
                content.append(y.text.strip())
    
    rows.append((url, title, date, " ".join(content).strip()))

import openpyxl # type: ignore
df = pd.DataFrame(rows)
path = "C://Users//JABAMI//Desktop//ours_Sorbonne_FTD//applied_data_science//rendu_final//database.xlsx"
df.to_excel(path)