#!/usr/bin/env python3
import json
import requests
import time
import re

def query(term, year, query_type="Journal_Articles", format="json"):
    url = "http://dblp.uni-trier.de/search/publ/api?q={term}%20type%3A{query_type}%3A%20year%3A{year}%3A&h=1000&format={format}".format(term=term, year=year, query_type=query_type, format=format)
    r = requests.get(url)
    return r.text


def query_impact(journal):
    url = "http://www.scijournal.org/search/search.php?search=1&x=6&y=15"
    r = requests.get(url, params={"query": journal})
    text = r.text
    p = text.find('class="title">	Impact Factor of')
    impact_url = ""
    if p >= 0:
        href = text.rfind('href="', 0, p)
        if href >= 0:
            href += 6
            end_href = text.find('"', href)
            impact_url = text[href:end_href]

    impact = 0
    if impact_url:
        # print(impact_url)
        r = requests.get(impact_url)
        text = r.text
        m = re.search(r"Impact Factor\s*:\s*([\d]+\.[\d]*)", r.text)
        if m:
            impact = float(m.group(1))
    return impact


def query_venue_info(venue, venue_type="journal"):
    text = ""
    url = "http://dblp.uni-trier.de/search/venue/api?h=1&format=json"
    r = requests.get(url, params={"q": venue})
    name = ""
    impact = 0
    if r.text:
        hits = json.loads(r.text)["result"]["hits"]
        if hits and "hit" in hits:
            name = hits["hit"][0]["info"]["venue"]

        if venue_type == "journal" and name:
            impact = query_impact(name)
    return name, impact


if __name__ == "__main__":
    term = "electrocardiogram|ECG|arrhythmia"
    query_types = (
        ("journal", "Journal_Articles"),
        ("conf", "Conference_and_Workshop_Papers")
    )
    for type_name, query_type in query_types:
        venues = {}
        rows = []
        for year in range(2006, 2016+1):
            print(query_type, year)
            text=query(term, year, query_type)
            hits = json.loads(text)["result"]["hits"]["hit"]
            for hit in hits:
                info = hit["info"]
                title = info["title"]
                year = info["year"]
                venue = info["venue"]
                url = info["url"]
                rows.append((year, venue, title, url))
                count = venues.setdefault(venue, 0)
                venues[venue] = count + 1
            time.sleep(2)

        venue_info = {}
        for venue in venues:
            name, impact = query_venue_info(venue, type_name)
            print(name, impact)
            venue_info[venue] = (name, impact)
            time.sleep(1)

        f = open("dblp_{0}_2006_2016.csv".format(type_name), "w")
        f.write('"year","venue","title","url","freq","IF","name"\n')
        for row in rows:
            venue = row[1]
            freq = venues[venue]
            name, impact = venue_info[venue]
            f.write(('"%s",' * len(row)) % row)
            f.write('%d,%f,"%s"\n' % (freq, impact, name))
        f.close()
