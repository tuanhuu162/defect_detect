import pandas as pd
import xml.etree.ElementTree as ET
import io

def etree_to_dict(t):
    d = {t.tag : map(etree_to_dict, t.iterchildren())}
    d.update(('@' + k, v) for k, v in t.attrib.iteritems())
    d['text'] = t.text
    return d

def iter_docs(author):
    author_attr = author.attrib
    for doc in author.iter('BugInstance'):
        print(doc.tag, doc.attrib)
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        yield doc_dict

xml_data = open("Benchmark_1.2beta-findsecbugs-v1.4.3-196.xml")

xml_label = pd.read_csv("Benchmark_v1.2_Scorecard_for_FBwFindSecBugs.csv")

etree = ET.parse(xml_data) #create an ElementTree object
# print(etree.getroot())
list_feature = list(iter_docs(etree.getroot()))
# print(list_feature)
# doc_df = pd.DataFrame(list(iter_docs(etree.getroot())))

# print(doc_df)