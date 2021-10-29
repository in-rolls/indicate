# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    User = scrapy.Field()
    Comment = scrapy.Field()
    Time = scrapy.Field()


class Fixture(scrapy.Item):

    unique_id = scrapy.Field()
    dropdown_value = scrapy.Field()
    party_name = scrapy.Field()
    name_english = scrapy.Field()
    name_hindi = scrapy.Field()
    assembly_constituency = scrapy.Field()
    state = scrapy.Field()
    application_uploaded = scrapy.Field()
    current_status = scrapy.Field()
    fathers_or_husbands_name_english = scrapy.Field()
    name = scrapy.Field()
    fathers_or_husbands_name_hindi = scrapy.Field()
    address = scrapy.Field()
    age = scrapy.Field()
    url = scrapy.Field()
    photo_url = scrapy.Field()
    photo_filename_with_unique_id = scrapy.Field()
    affidavit_filename_with_unique_id = scrapy.Field()
