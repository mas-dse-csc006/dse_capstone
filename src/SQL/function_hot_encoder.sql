CREATE OR REPLACE FUNCTION hotencoder (item_array text[],full_list text[])
RETURNS text AS $$
results = [ '1' if i in item_array else '0' for i in full_list]
results = ''.join(results)
return results
$$ LANGUAGE plpython3u;

/*
with q as (
select '{a,c,d,g,j}'::text[] x, '{r,q,a,b,c,d,y,e,f,i,w}'::text[] y
)
select hotencoder(x,y) from q 
;
*/
