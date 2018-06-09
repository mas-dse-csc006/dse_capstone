\set infile :datadir '/in_asin_cat_tags_level4.csv'
COPY product_category 
FROM :'infile'
WITH DELIMITER E'\t' CSV HEADER;

reindex table product_category ;
