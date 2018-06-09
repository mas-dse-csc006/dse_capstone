






select * from review_mv limit 10 ;


-- limit 2 ;
) to 
'/Users/nolanthomas/Public/amazon/out_cat_ptile_season_rating_polarity_sentiment.csv' 
DELIMITER ',' CSV HEADER;

copy (
select 
    reviewerid,
    asin,
    brand,
    price,
    p0_25,
    p25_50,
    p50_75,
    p75_95,
    p95_99,
    p99_100
from review_details_v
) to 
'/Users/nolanthomas/Public/amazon/out_asin_brand_price_ptiles.csv' 
DELIMITER ',' CSV HEADER;


--reviewerid, unixtime, price, brand,  from 
--review_details_v limit 5 
;

-- 181,329
   
select count(distinct json_reviews_clothing_5core.data ->> 'reviewerID'::text) from json_reviews_clothing_5core ;
-- 278,677
-- 39,387 reviewerids

select  count(distinct reviewid) from reviews_women_scraped_cpp_fv limit 5;
-- 181,329
-- '35,358' (reviewid)

select price::numeric from product_mv ; --where price is not null limit 1;


create table cat_clothing_5core as
with q as (
    select count(*) c, asin from unfold_reviews_clothing_5core
    group by asin
)

select sum(c) c, category, category_key, 'c >= 25' clause, current_time load_date
from q
inner join asin_category l on q.asin = l.asin
--where category not in ('Clothing, Shoes & Jewelry','Women') and c >= 25
group by category, category_key, clause, load_date
    order by c desc
;

select count(*) from cat_clothing_5core where category not in ('Clothing, Shoes & Jewelry','Women') and c >= 25
-- 847

select * from asin_category limit 5;

with a as (
select asin from asin_category where category in ('Women', 'Clothing, Shoes & Jewelry') group by asin 
having count(asin) = 2
)
select count(*) from unfold_reviews_clothing_5core r
inner join a on r.asin = a.asin;
-- 180612



---------

with q as (
    select count(*) c, asin from reviews_women_scraped_cpp_fv
    group by asin
),
z as (
select sum(c) c, category, category_key, 'c >= 25' clause, current_time load_date
from q
inner join asin_category l on q.asin = l.asin
where category not in ('Clothing, Shoes & Jewelry','Women') and c >= 25
group by category, category_key, clause, load_date
) select count(*) from z ;

---863 with all

select
  percentile_cont(0.25) within group (order by price asc) as percentile_25,
  percentile_cont(0.50) within group (order by price asc) as percentile_50,
  percentile_cont(0.75) within group (order by price asc) as percentile_75,
  percentile_cont(0.95) within group (order by price asc) as percentile_95,
  percentile_cont(0.99) within group (order by price asc) as percentile_99
from  reviews_women_scraped_cpp_fv
;

select * from reviews_women_scraped_cpp_fv where price > 166.63;




copy (
select 
asin,
reviewerid,
title,
overall,
image_url
from review_details_v)
to 
'/Users/nolanthomas/Public/amazon/out_review_details_v.csv' 
DELIMITER ',' CSV HEADER;

with s as (
	select 0.25 polarity 
	union
    select -0.25
    union 
    select -0.75
    union
    select 0.90
    union
    select 0.05
    union 
    select -0.01
    union
    select 0.5
    union 
    select -0.5
    union
    select -0.1
    union
    select 0.1
)
select
polarity,
case when polarity < -0.5 then 1 else 0 end very_negative,
case when polarity >= -0.5 and polarity < -0.1 then 1 else 0 end negative,
case when polarity >= -0.1 and polarity < 0.1 then 1 else 0 end as netural, 
case when polarity >= 0.1 and polarity < 0.5 then 1 else 0 end positive,
case when polarity >= 0.5 then 1 else 0 end very_positive
from s
order by polarity;


select 
sum(very_negative),
sum(negative),
sum(neutral),
sum(postive),
sum(very_positive)

