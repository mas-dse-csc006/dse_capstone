CREATE OR REPLACE FUNCTION get_categories(cat_text text)
RETURNS setof text AS $$

import ast
#cat_text = '["CDs & Vinyl", "Classical"], ["CDs & Vinyl", "Jazz", "Swing Jazz", "Classic Big Band"], ["CDs & Vinyl", "Jazz", "Swing Jazz", "Contemporary Big Band"], ["CDs & Vinyl", "Pop", "Easy Listening"], ["CDs & Vinyl", "Pop", "Vocal Pop"], ["Musical Instruments", "Instrument Accessories", "General Accessories"]'
#cat_text ='[]'
#cat_text = None
results = set()
if cat_text is not None and len(cat_text) > 0:
    cat_parsed = ast.literal_eval(cat_text)
    results = set(i for c in cat_parsed for i in c)
return results

$$ LANGUAGE plpython3u;