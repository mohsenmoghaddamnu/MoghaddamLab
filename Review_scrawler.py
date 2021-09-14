import selenium
import time
from selenium import webdriver
from bs4 import BeautifulSoup as BS
from selenium.webdriver.chrome.options import Options
import sys
import pickle
import pandas as pd
from tqdm import tqdm, tqdm_notebook





def Scrawl_product_description(read_from = 'str'):
    with open(read_from, "r") as x:
        all_product = x.read().splitlines()

    product_name_description = {}
    for name in tqdm([all_product]):
        for url in name:
            product = {}
            options = Options()
            options.add_argument("start-maximized")
            browser = webdriver.Chrome(chrome_options=options,
                                       executable_path='/home/ramin/Downloads/chromedriver_linux64/chromedriver')
            product_name = url.split('/')[3]
            style_id = all_product[0].split('/')[4].split('?')[1].split('&')[0].split('=')[1]
            color_id = all_product[0].split('/')[4].split('?')[1].split('&')[1].split('=')[1]
            product_id = all_product[0].split('/')[4].split('?')[0]
            pid = product_id + "_" + style_id
            browser_name = "https://www.finishline.com" + url
            browser.get(browser_name)

            bs_obj = BS(browser.page_source.encode('utf-8'), "html.parser")
            description = bs_obj.findAll("div", {"class":"column small-12"}) 
            for i in description:
                details = ",".join(i.stripped_strings)
                browser.quit()
            product['style_id'] = style_id
            product['color_id'] = color_id
            product['description'] = details
            product['product_id'] = product_id

            if product_name in product_name_description:
                product_name_description[product_name].append(product)
            else:
                product_name_description[product_name] = [product]
    return product_name_description
    