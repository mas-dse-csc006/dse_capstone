DROP INDEX IF EXISTS json_product_asin_idx ;
DROP INDEX IF EXISTS json_reviews_e_5_idx ;
DROP INDEX IF EXISTS json_reviews_csj_5_idx ;
DROP INDEX IF EXISTS json_reviews_e_5_idx2 ;
DROP INDEX IF EXISTS json_reviews_csj_5_idx2 ;

CREATE INDEX json_product_asin_idx ON json_product USING BTREE ((data->>'asin'));
CREATE INDEX json_reviews_e_5_idx ON json_reviews_electronics_5 USING BTREE ((data->>'reviewerID'));
CREATE INDEX json_reviews_csj_5_idx ON json_reviews_clothing_shoes_and_jewelry_5 USING BTREE ((data->>'reviewerID'));
CREATE INDEX json_reviews_e_5_idx2 ON json_reviews_electronics_5 USING BTREE ((data->>'asin'));
CREATE INDEX json_reviews_csj_5_idx2 ON json_reviews_clothing_shoes_and_jewelry_5 USING BTREE ((data->>'asin'));
