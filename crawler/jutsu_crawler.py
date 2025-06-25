# import scrapy

# class BlogSpider(scrapy.Spider):
#     name = 'narutospider'
#     start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

#     def parse(self, response):
#         for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
#             extracted_data = scrapy.Request("https://naruto.fandom.com"+href
#                                             callback=self.parse_jutsu)
#             yield extracted_data

#         for next_page in response.css('a.w-nextlink'):
#             yield response.follow(next_page, self.parse)