#!/usr/bin/env python3
import json
import requests
import time
import re
import difflib
import google  # python google 1.8 package


def query_dblp(term, year, query_type="Journal_Articles", format="json"):
    url = "http://dblp.uni-trier.de/search/publ/api?q={term}%20type%3A{query_type}%3A%20year%3A{year}%3A&h=1000&format={format}".format(term=term, year=year, query_type=query_type, format=format)
    r = requests.get(url)
    return r.text


def extact_doi_url(dblp_article_url):
    doi_url = ""
    r = requests.get(dblp_article_url)
    if r.text:
        m = re.search(r'href="(http://dx.doi.org/[^"]+)"', text)
        if m:
            doi_url = m.group(1)
    return doi_url


# query impact factor from Google + ResearchGate website
last_google_time = 0
def query_research_gate_impact(journal):
    impact = 0

    # avoid calling google too often
    delta_time = time.time() - last_google_time
    if delta_time < 2.0:
        time.sleep(2.0 - delta_time)
        last_google_time = time.time()

    for url in google.search("researchgate {0}".format(journal), stop=1):
        r = requests.get(url)
        text = r.text
        if text:
            m = re.search(r"Current impact factor:\s*([\d]+\.[\d]*)", text)
            if m:
                impact = float(m.group(1))
        break
    return impact


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
        m = re.search(r"Impact Factor\s*:\s*([\d]+\.[\d]*)", r.text)
        if m:
            impact = float(m.group(1))
    return impact


def query_venue_info(venue, venue_type="journal"):
    text = ""
    url = "http://dblp.uni-trier.de/search/venue/api?h=1000&format=json"
    r = requests.get(url, params={"q": venue})
    name = ""
    impact = 0
    if r.text:
        hits = json.loads(r.text)["result"]["hits"]
        if hits and "hit" in hits:
            hits = hits["hit"]
            # get all similar venue names from dblp database
            venues = [hit["info"]["venue"] for hit in hits]

            # find closest match
            venue = venue.replace(".", "")  # remove all dots
            matches = difflib.get_close_matches(venue, venues, 1, 0.0)
            if matches:
                name = matches[0]
                name = name.replace("&amp;", "&")

        if venue_type == "journal" and name:
            impact = query_impact(name)
            if not impact:
                # try to quote the name
                impact = query_impact('"{0}"'.format(name))
                if not impact:
                    # try alternative names (replace : with -)
                    impact = query_impact(name.replace(": ", "-"))
                    if not impact:
                        # last resort, try google search + research gate
                        impact = query_research_gate_impact(name)
    return name, impact


if __name__ == "__main__":
    begin_year = 2006
    end_year = 2016
    query_types = (
        ("journal", "Journal_Articles"),
        ("conf", "Conference_and_Workshop_Papers")
    )
    query_terms = ("arrhythmia", "atrial.fibrillation", "atrial.flutter", "ventricular.fibrillation", "tachycardia")
    for type_name, query_type in query_types:
        venues = {}
        rows = []
        for year in range(begin_year, end_year + 1):
            for term in query_terms:
                print(term, query_type, year)
                text = query_dblp(term, year, query_type)
                hits = json.loads(text)["result"]["hits"]
                if hits and "hit" in hits:
                    hits = hits["hit"]
                    for hit in hits:
                        info = hit["info"]
                        title = info["title"]
                        year = info["year"]
                        venue = info["venue"]
                        url = info["url"]
                        if url:  # query the doi link to the document
                            time.sleep(0.5)
                            doi_url = extact_doi_url(url)
                            if doi_url:
                                url = doi_url

                        row = (year, title, venue, url)
                        # this is inefficient but inevitable (we may lose the sort order if we use set)
                        if row not in rows:  # avoid duplications
                            rows.append(row)
                        count = venues.setdefault(venue, 0)
                        venues[venue] = count + 1
                time.sleep(1.0)

        venue_info = {}
        for venue in venues:
            name, impact = query_venue_info(venue, type_name)
            print(venue, "=>", name, ": ", impact)
            venue_info[venue] = (name, impact)
            time.sleep(1.0)

        f = open("dblp_{0}_{1}_{2}.csv".format(type_name, begin_year, end_year), "w")
        f.write('"year","title","venue","url","freq","IF","name"\n')
        for row in rows:
            venue = row[2]
            freq = venues[venue]
            name, impact = venue_info[venue]
            f.write(('"%s",' * len(row)) % row)
            f.write('%d,%f,"%s"\n' % (freq, impact, name))
        f.close()
