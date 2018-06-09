\set infile :datadir '/in.strict.reviews_Electronics_5.json'
COPY json_reviews_electronics_5 (data) FROM :'infile'
csv quote e'\x01' delimiter e'\x02'
;
