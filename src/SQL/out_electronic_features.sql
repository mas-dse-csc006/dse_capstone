\set outfile :datadir '/v2.out_electronic_features.tab.csv'
copy (
select * from electronic_features_v
) 
TO :'outfile'
DELIMITER E'\t' CSV HEADER;

