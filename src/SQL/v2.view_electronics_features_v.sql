drop view if exists electronic_features_v ;
create or replace view electronic_features_v as 
with l4 as (
select distinct level4_electronics level4 from product_category_v where  electronics_flag = 'Y' 
),
/* all level4's to support hotencoding */
l4_list as (
select array_agg(distinct level4 order by level4) level4_list 
from l4), 
tag_list as (
select array_agg(distinct tag order by tag) tag_list 
    from 
    category_tag_v 
    where 
    asin_count > 1 and electronics_flag = 'Y'
),
rating as (
  select avg((j.data->>'overall')::numeric) average_rating, j.data->>'asin'::text asin 
    from json_reviews_electronics_5 j 
    group by j.data->>'asin'::text
)
select
reviewerid as "reviewerID",
v.asin,
unixtime as "unixReviewTime",
brand brand,
price,
'''' feature_vector,
'''' || hotencoder(category_tags,tag_list) || '''' as top_categories,    
'''' || hotencoder(('{'|| level4 ||'}')::text[],level4_list) || '''' as level4,
level4 level4_name,
average_rating,
l4_avg_adjusted price_delta_l4avg,
level4_average
from electronics_review_details_v v
inner join rating on rating.asin = v.asin
    cross join l4_list
    cross join tag_list
--limit 5000 
;
