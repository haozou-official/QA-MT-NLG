import pandas as pd
import string
import os
import re
import urllib.request
import gzip
import shutil
#import rdfxml
import json
import urllib

service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
query = [{ "name": "citigroup", "/common/topic/alias": []}]

params = { 'query': json.dumps(query), 'limit':5}   #'key': key,

url = service_url + '?' + urllib.parse.urlencode(params)
response = json.loads(urllib.request.urlopen(url).read())

zipfile_path = '/content/gdrive/Shareddrives/Improving-QA-MT/Colab/Answer_Aliasing/freebase-rdf-latest.gz'
save_path = '/content/gdrive/Shareddrives/Improving-QA-MT/Colab/Answer_Aliasing/freebase_data.txt'

with gzip.open(zipfile_path, 'rb') as f_in:
    with open(save_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
sparql.setQuery("""
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q146 .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
results_df = pd.io.json.json_normalize(results['results']['bindings'])

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery("""
    PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT ?p ?pt ?pLabel ?d ?aliases WHERE {
  {
    SELECT ?p ?pt ?d
              (GROUP_CONCAT(DISTINCT ?alias; separator="|") as ?aliases)
    WHERE {
      ?p wikibase:propertyType ?pt .
      OPTIONAL {?p skos:altLabel ?alias FILTER (LANG (?alias) = "en")}
      OPTIONAL {?p schema:description ?d FILTER (LANG (?d) = "en") .}
    } GROUP BY ?p ?pt ?d
  }
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
  }
}""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    print(result)
    #print(result["label"]["value"])

# importing the module
import wikipedia

# getting suggestions
result = wikipedia.search("obama", results = 5)

#Timothy Donald Cook
# printing the result
print(result)
