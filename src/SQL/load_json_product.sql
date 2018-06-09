\set infile :datadir '/in.strict.metadata.json'
COPY json_product (data) FROM :'infile'
csv quote e'\x01' delimiter e'\x02'
;
