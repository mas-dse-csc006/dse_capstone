drop table sharknado_women ;
create table sharknado_women (
reviewid varchar(50),
asin varchar(50),
rating decimal,
unixtime integer,
price decimal,
brand varchar(100)
    );
    
    
COPY sharknado_women FROM '/Users/nolanthomas/Public/amazon/reviews_Women.csv' 
WITH DELIMITER ',' CSV HEADER;

select count(*) from sharknado_women;

select count(*) from product ;
select * from reviews_clothing_5core limit 5;

select count(distinct asin) from sharknado_women ;

select count(distinct asin) from review_mv;

