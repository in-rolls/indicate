from scrapy import Spider, Request
from scrapy.selector import Selector
from crawler.items import CrawlerItem, Fixture
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import re



class CrawlerSpider(Spider):
    name = "crawler"
    allowed_domains = ["affidavit.eci.gov.in"]
    start_urls = [
        "https://affidavit.eci.gov.in",
    ]
    rules = [
        Rule(LinkExtractor(allow=r'show-profile'), callback='parse_product'),
    ]
    unique_id = 0

    def parse(self, response):
        # questions = Selector(response).xpath('//ul[@class="table"]/li')
        questions = response.selector
        print ('11111111111111111111', questions)
        # get next page
        for p in questions.css(".pagination"):
            detail_link = p.xpath('li[@class="page-item"]')
            # print ('------------------', detail_link)
            for link in detail_link:
                n = link.extract()
                # print ('_____________________',n)
                matches = re.findall('rel="next"', n)
                # print(matches)
                if matches:
                    next_link = link.xpath('a/@href').extract()
                    print ('kakakakakakak', next_link)
                    yield Request(next_link[0], callback=self.parse)

        # item['name_english'] = tr.xpath('td[2]/div[@class="details-name"]/h4/text()').extract()

        for tr in questions.css("table#data-tab>tbody>tr"):
            item = Fixture()
            detail_link = tr.xpath('td[1]/div[@class="img-bx"]/a/@href').extract()
            # print ('=======================', detail_link)
            item['name_english'] = tr.xpath('td[2]/div[@class="details-name"]/h4/text()').extract()
            # item['link'] =detail_link
            # yield item
            yield Request(detail_link[0], callback=self.parse_product)

    def parse_product(self, response):
        # questions = Selector(response).xpath('//ul[@class="table"]/li')
        questions = response.selector
        print ('2222222222222222', questions)
        # for sec in questions.css('section'):
        #     print ('3333333333333333',sec)
        can_details = questions.css('section')[0]
        # can_per_details = questions.css('section')[1]
        can_details = can_details.xpath('div/div/div/div/div[@class="card-body"]/div/div')
        can_details = can_details[1].xpath('div')
        # name = can_details[3]

        item = Fixture()
        self.unique_id +=1
        item['unique_id'] = self.unique_id
        item['party_name'] = can_details[0].xpath('div[2]/text()').extract()[0].strip('\n').strip()
        item['name_english'] = can_details[2].xpath('div[2]/div/p/b/text()').extract()[0].strip('\n').strip()
        item['name_hindi'] = can_details[3].xpath('div[2]/div/p/b/text()').extract()[0].strip('\n').strip()
        item['assembly_constituency'] = can_details[4].xpath('div[2]/div/p/text()').extract()[0].strip('\n').strip()
        item['state'] = can_details[5].xpath('div[2]/div/p/text()').extract()[0].strip('\n').strip()
        item['application_uploaded'] = can_details[6].xpath('div[2]/div/p/text()').extract()
        item['current_status'] = can_details[7].xpath('div[2]/div/p/text()').extract()

        can_per_details = questions.css('section')[1]
        can_per_details = can_per_details.xpath('div/div[@class="row"]/div/div/div/div/div/form/div/div[1]/div')

        item['fathers_or_husbands_name_english'] = can_per_details[0].xpath('div/p/text()').extract()
        item['fathers_or_husbands_name_hindi'] = can_per_details[1].xpath('div/p/text()').extract()
        item['name'] = can_per_details[2].xpath('div/p/text()').extract()

        item['address'] = can_per_details[4].xpath('div/p/text()').extract()
        item['age'] = can_per_details[6].xpath('div/p/text()').extract()
        # item['url'] = can_per_details[0].xpath('div[2]/text()').extract()[0].strip('\n').strip()
        # item['photo_url'] = can_per_details[0].xpath('div[2]/text()').extract()[0].strip('\n').strip()
        # item['photo_filename_with_unique_id'] = can_per_details[0].xpath('div[2]/text()').extract()[0].strip('\n').strip()
        # item['affidavit_filename_with_unique_id'] = can_per_details[0].xpath('div[2]/text()').extract()[0].strip(
         #'\n').strip()

        yield item
