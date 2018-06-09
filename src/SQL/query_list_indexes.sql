select
    t.relname as table_name,
    i.relname as index_name,
    a.attname as column_name
from
    pg_class t,
    pg_class i,
    pg_index ix,
    pg_attribute a
where
    t.oid = ix.indrelid
    and i.oid = ix.indexrelid
    and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey)
    and t.relkind = 'r'
    and t.relname in ('product_mv', 
                      'review_mv',
                      'reviewer_asin_crossed',
                      'json_product',
                      'product_womens',
                      'reviews_women_scraped_cpp_fv',
                      'json_reviews_clothing_5core',
                      'unfold_reviews_clothing_5core')
order by
    t.relname,
    i.relname;

reindex index  product_womens_asin_idx;
reindex index  reviews_women_scraped_cpp_fv_idx1;
reindex index reviews_women_scraped_cpp_fv_idx2;
reindex index reviews_women_scraped_cpp_fv_idx3;