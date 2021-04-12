# Imports
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
import time

# Set driver for Chrome
options = webdriver.ChromeOptions()
options.add_argument('-headless')
options.add_argument('-no-sandbox')
options.add_argument('-disable-dev-shm-usage')


def scrape_tiki(url):

    url_pattern = r'src=*.*'
    url_replace = 'page='
    new_url = re.sub(url_pattern,url_replace,url)

    # List containing data of all products
    data = []
    page_num = 1

    while True:
        # Get parsed HTML
        soup = get_soup(page_num, new_url)

        

        # Find all article tags
        products = soup.body.find_all('a',{'class':'product-item'})

        #Break Condition
        if len(products) == 0:
          break

        page_num += 1

        i = 0
        product_pattern = r'\w\d+\.'


        for i in range(len(products)-1):

            # Dictionary:
            d = {}
            i += 1

            # Get outside tags
            products_style = products[i].span
            products_info = products_style.find('div', {'class':'info'})
            badge = products_info.find('div', {'class':'badge-benefits'})
            service = products_info.find('div',{'class': 'badge-service'})
            
            try:
              
              # ID
              d['product_id'] = re.search(product_pattern, products[i]['href']).group().strip('\.p')

              # Title
              d['product_title'] = products_info.find('div', {'class':'name'}).text

              # Price
              d['price'] = products_info.find('div', {'class':'price-discount'}).div.text

              # Number of reviews
              if products_info.find('div', {'class':'review'}):
                d['number_of_review'] = products_info.find('div', {'class':'review'}).text[1:-1]
              else:
                d['number_of_review'] = 'Unavailable'

              # Rating average
              if products_info.find('div', {'class':'rating__average'}):
               d['percentage_of_star'] = products_info.find('div', {'class':'rating__average'})['style'].strip('width: ')
              else:
               d['percentage_of_star'] = 'Unavailable'  

              # Product Discount
              if products_info.find('div',{'class':'price-discount__discount'}):
                d['discount'] = products_info.find('div',{'class':'price-discount__discount'}).text
              else:
                d['discount'] = 'Unavailable'

              if products_info.find('div', {'class':'freegift-list'}):
                d['free-gift'] = 'Available'
              else:
                d['free-gift'] = 'Unavailable'

              # URLs
              d['image_url'] = products[i].span.div.div.img['src']
              d['product_page_url'] = 'https://tiki.vn' + products[i]['href']

              #Cheaper Than Return Status
              if products_info.find('div', {'class':'badge-under-price'}).div:
                d['Cheaper_Than_Return'] = "Available"
              else:
                d['Cheaper_Than_Return'] = "Unavailable"

              # Installment Payment Status
              if badge.div:
                d['Installment_payment'] = "Available" 
              else:
                d['Installment_payment'] = "Unavailable"

              # Tikinow Status
              if service.find('div',{'class': 'item'}):
                d['tikinow'] = 'Available'
              else:
                d['tikinow'] = 'Unavailable'

              # Free Delivery Status
              if products[i].find('div', {'class':'item top'}):
               d['free_delivery'] = 'Available'
              else:
               d['free_delivery'] = 'Unavailable'

              
              # Append the dictionary to data list
              data.append(d)

            except:
              # Skip if error and print error message
              print("We got one article error!")
    return data
def get_soup(page_num, new_url):
      time.sleep(5)   
      tiki_url = new_url + str(page_num)

      # Set up driver
      driver = webdriver.Chrome('chromedriver',options=options)
      driver.implicitly_wait(30)
      driver.get(tiki_url)

      html_data = driver.page_source 
      soup = BeautifulSoup(html_data, 'html.parser')

      return soup


# Final Data
data = scrape_tiki('https://tiki.vn/dien-thoai-smartphone/c1795?src=static_block')
product_category = pd.DataFrame(data = data, columns = data[2].keys())
product_category.to_csv("./result.csv", index=False, encoding='utf-8')