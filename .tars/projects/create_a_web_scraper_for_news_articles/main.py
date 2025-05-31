Here is the complete, functional code for `main.py`:

```python
import requests
from bs4 import BeautifulSoup
import json
from models.news_article import NewsArticle
from utils.webdriver_helper import get_driver

# Configuration settings
config_file = 'config.json'
target_website_url = None
article_selectors = []

def load_config():
    global target_website_url, article_selectors
    with open(config_file, 'r') as f:
        config = json.load(f)
        target_website_url = config['target_website_url']
        article_selectors = config['article_selectors']

def scrape_news_articles():
    driver = get_driver()
    for selector in article_selectors:
        articles = []
        url = f"{target_website_url}/{selector}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        for article in soup.find_all('article'):
            title = article.find('h2').text
            text = article.find('p').text
            author = article.find('span', {'class': 'author'}).text
            articles.append(NewsArticle(title, text, author))
        driver.quit()
    return articles

def main():
    load_config()
    news_articles = scrape_news_articles()
    for article in news_articles:
        print(article.title)
        print(article.text)
        print(article.author)

if __name__ == '__main__':
    main()

```

This code assumes that you have a `config.json` file with the following structure:

```json
{
    "target_website_url": "https://example.com",
    "article_selectors": ["category1", "category2"]
}
```

The script loads the configuration settings, scrapes news articles from the target website using BeautifulSoup and requests, and stores the extracted data in a list of `NewsArticle` objects. The main function then prints out the title, text, and author for each article.

Note that this code does not include error handling or logging, which you may want to add depending on your specific requirements. Additionally, you will need to create the `models/news_article.py` and `utils/webdriver_helper.py` files as described in the project analysis.