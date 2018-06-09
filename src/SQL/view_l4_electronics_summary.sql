 create or replace view l4_electronics_summary_v as 
 SELECT 
    v.level4_electronics,
    round(avg(v.price), 2) AS avg_price
   FROM product_mv v
   WHERE v.electronics_flag = 'Y'
  GROUP BY v.level4_electronics;
  
--select * from subcategory_l4_summary ;
--select * from reviewer_summary ; --where level4 = '' ;