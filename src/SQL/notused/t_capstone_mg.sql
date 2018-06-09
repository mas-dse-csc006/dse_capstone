drop table women ;
create table women as 
select 
r.reviewid,
r.asin,
r.rating,
r.price,
r.brand,
TIMESTAMP 'epoch' + r.unixtime * INTERVAL '1 second' as reviewtime,
p.data->'categories' categories,
p.data->>'price' product_price,
p.data->>'brand' product_brand,
p.data->'related'->'also_viewed' also_viewed,
p.data->'related'->'also_bought' also_bought,
p.data->'related'->'bought_together' bought_together
from product_mv p
inner join sharknado_women r on p.asin = r.asin
--limit 10
;

create index sharknado_w_idx on sharknado_women(asin);

drop materialized view product_mv ;
create materialized view product_mv as 
select 
p.data->>'asin' asin,
p.data->>'categories' categories,
p.data->>'price' price,
p.data->>'brand' brand,
p.data->'related'->'also_viewed' also_viewed,
p.data->'related'->'also_bought' also_bought,
p.data->'related'->'bought_together' bought_together,
p.data
from product p
;
create index product_mv_idx on product_mv(asin) ;

select p.data->'related'->'also_viewed', p.data->'asin' from product p limit 1 ;


select * from women ;

select count(*) from women ;

select count(*) from sharknado_women ;


COPY (SELECT row_to_json(t) FROM women as t)
to '/Users/nolanthomas/Public/amazon/out_women.json';

COPY (SELECT  FROM women as t)
to '/Users/nolanthomas/Public/amazon/out_women.json';