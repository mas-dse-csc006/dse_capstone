drop table if exists product_category ;
create table product_category (
    asin varchar(50),
    categories varchar(2000),
    category_tags varchar(1000),
    level4_women varchar(50),
    level4_electronics varchar(50)
);
create index product_category_asin_idx on product_category( asin ) ;
