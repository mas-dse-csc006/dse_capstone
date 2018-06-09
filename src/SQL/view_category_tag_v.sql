drop materialized view if exists category_tag_v ;
create materialized view category_tag_v as
	with tag as (
    select 
    asin,
    unnest(category_tags) tag,
    women_flag,
    electronics_flag
    from product_category_v
    )
    select 
    count(asin) asin_count,
    tag,
    women_flag,
    electronics_flag
    from tag
    group by 
    tag, women_flag, electronics_flag
    ;   

