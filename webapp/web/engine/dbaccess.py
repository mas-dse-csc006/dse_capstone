import psycopg2
import numpy as np


class DBUtil(object):
    def __init__(self):
        super(DBUtil, self).__init__()
        # self.hostname = 'ec2-35-173-188-150.compute-1.amazonaws.com'
        self.hostname = 'localhost'
        self.port = 5433
        self.user = 'other_user'
        self.password = 'capstone!'
        self.dbname = 'postgres'
        self.conn = psycopg2.connect(
            host=self.hostname, user=self.user, password=self.password, dbname=self.dbname, port=self.port)

    def get_amazon_user(self, email):
        cur = self.conn.cursor()
        strSQL = """
            select 
                a.reviewerid, b.asin
            from amazon_reviewer a
            left join review b
            on a.reviewerid = b.reviewerid
            where a.username = '{0}'
        """.format(email)

        cur.execute(strSQL)

        return cur.fetchall()

    def get_product_item_random(self, limit=10):
        cur = self.conn.cursor()

        strSQL = """
            select
                asin,
                level4,
                price,
                brand,
                title,
                image_url
             from product order by random() limit {0};
        """.format(limit)

        cur.execute(strSQL)
        return cur.fetchall()

    def breakdown_categories(self, asin_list):
        cur = self.conn.cursor()
        list = "({0})".format(
            ','.join(["'{0}'".format(asin) for asin in asin_list]))
                    
        strSQL = """        
            select level4, count(level4) from product
            where asin in {0}
            group by level4        
        """.format(list)

        print strSQL

        cur.execute(strSQL)
        return cur.fetchall()

    def get_product_item(self, asin_list):
        cur = self.conn.cursor()
        list = "({0})".format(
            ','.join(["'{0}'".format(asin) for asin in asin_list]))

        strSQL = """
            select asin, title, image_url, price from product
            where asin in {0};
        """.format(list)

        print strSQL

        cur.execute(strSQL)
        return cur.fetchall()

    def get_bpr_recommended_item(self, asin_list, category, limit):
        cur = self.conn.cursor()
        list = "({0})".format(
            ','.join(["'{0}'".format(asin) for asin in asin_list]))

        strSQL = """
            select 
                asin,
                level4,
                price,
                brand,
                title,
                image_url
            from product
            where asin in {0} and level4 = '{1}' order by random() limit {2};
        """.format(list, category, limit)

        print strSQL

        cur.execute(strSQL)
        return cur.fetchall()        

    def get_random_category(self, limit):
        cur = self.conn.cursor()

        strSQL = """
            select * from
            (select distinct level4 from product
            group by level4) a
            order by random() limit {0};
        """.format(limit)

        cur.execute(strSQL)
        return cur.fetchall()

    def get_random_product_by_category(self, category, limit):
        cur = self.conn.cursor()

        strSQL = """
            select
                asin,
                level4,
                price,
                brand,
                title,
                image_url
            from product 
            where level4 = '{0}'
            order by random()            
            limit {1};
        """.format(category, limit)

        cur.execute(strSQL)
        return cur.fetchall()    
    
    def search(self, keywords, limit):
        cur = self.conn.cursor()

        strSQL = """
            select
                asin,
                level4,
                price,
                brand,
                title,
                image_url
            from product 
            where level4 like '%{0}%' 
            OR categories like '%{0}%'
            OR brand like '%{0}%'
            OR title like '%{0}%'
            OR asin like '%{0}%'
            order by random()            
            limit {1};
        """.format(keywords, limit)

        cur.execute(strSQL)
        return cur.fetchall() 
   


      

      

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    db = DBUtil()
    # resultset = db.get_product_item(['B005GYGD7O','B005LERHD8'])
    # print resultset
    #
    # resultset = db.get_product_item_random()
    # for x in resultset:
    #     print x
    email = 'toby@mail.com'
    u = db.get_amazon_user(email)
    if len(u) > 0:
        print u
    else:
        print "no user"
    db.close()
