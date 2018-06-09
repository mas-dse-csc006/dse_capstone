\set infile :datadir '/in.strict.reviews_Clothing_Shoes_and_Jewelry_5.json'
COPY json_reviews_clothing_shoes_and_jewelry_5 (data) FROM :'infile'
csv quote e'\x01' delimiter e'\x02'
;
