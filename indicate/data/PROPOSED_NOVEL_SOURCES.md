# Proposed Novel Data Sources for South Asian Language Transliteration

**STATUS: PLANNING DOCUMENT - NOT IMPLEMENTED**

This document outlines potential novel data sources for transliteration training, following the successful approach of mining ESPN Cricinfo and election affidavits. These are **proposals that require implementation and validation**.

## Philosophy

Focus on **proper nouns** (people, places, organizations) from public sources that contain both native script and romanization. These sources should be:
- Publicly accessible
- High quality (official or community-verified)
- Underexploited for NLP research
- Scalable (thousands to millions of pairs)

## Proposed Sources (Require Implementation)

### 1. Wikidata Entity Labels
**Type**: Mixed proper nouns (people, places, organizations, culture)
**Potential Scale**: 1-5M pairs across 22 languages
**License**: CC0 (Public Domain)

**Concept**: Query Wikidata's structured multilingual labels via SPARQL
- Example: Q9535 has label "शाहरुख़ ख़ान" (hi) and "Shah Rukh Khan" (en)
- Entity types: Persons (Q5), Places (Q515, Q486972), Films (Q11424), etc.

**Implementation Needed**:
- SPARQL query construction
- API rate limiting
- Result pagination
- Entity type filtering

**Validation Required**:
- Test with small samples
- Verify data quality
- Check API reliability

---

### 2. Wikipedia Interwiki Links
**Type**: Proper nouns (article titles)
**Potential Scale**: 500K-2M pairs across 15+ languages
**License**: CC BY-SA

**Concept**: Mine article titles across Wikipedia language editions
- Example: "मुंबई" (Hindi Wikipedia) ↔ "Mumbai" (English Wikipedia)
- Focus on biographical, geographic, and cultural articles

**Implementation Needed**:
- Wikipedia API integration
- Interwiki link extraction
- Category filtering for proper nouns
- Deduplication logic

**Validation Required**:
- Test with known good articles
- Verify interwiki alignment accuracy
- Handle disambiguation pages

---

### 3. OpenStreetMap Place Names
**Type**: Geographic proper nouns
**Potential Scale**: 500K+ place names
**License**: ODbL

**Concept**: Extract multilingual name:* tags from OSM
- Example: name:ta="சென்னை", name:en="Chennai"
- Covers cities, towns, villages, landmarks, streets

**Implementation Needed**:
- Overpass API queries
- Tag filtering and extraction
- Geographic bounding boxes
- Rate limiting

**Validation Required**:
- Test queries on small regions
- Verify tag quality
- Check coverage

---

### 4. Indian Railway Station Names
**Type**: Place names (stations)
**Potential Scale**: 7,000+ stations across 10 languages
**Quality**: Government official

**Concept**: Extract official bilingual station names
- Example: "नई दिल्ली" (Hi) / "New Delhi" (En) - Code: NDLS
- Sources: Indian Railways data, Wikidata station entities

**Implementation Needed**:
- Identify authoritative data sources
- Parse station lists
- Map station codes to multilingual names
- Verify romanization standards

**Validation Required**:
- Cross-reference with official railway data
- Verify against IRCTC website
- Test with known major stations

---

### 5. Electoral Rolls (High Potential, High Complexity)
**Type**: Person names (voters)
**Potential Scale**: Millions of names per state
**Quality**: Government official
**Privacy Considerations**: HIGH

**Concept**: Parse voter names from state electoral roll PDFs
- Example: Column 1: "முத்து குமார்" (Tamil), Column 2: "Muthu Kumar" (English)
- Available from Chief Electoral Officer websites for each state

**Implementation Needed**:
- PDF download automation
- PDF table parsing (pdfplumber/PyPDF2)
- Column alignment and name extraction
- State-specific format handling
- Privacy-preserving anonymization

**Validation Required**:
- Test with sample PDFs from multiple states
- Verify name extraction accuracy
- Validate against manual inspection
- Legal/ethical review

**Privacy & Ethics**:
- Electoral rolls are public documents
- Remove all personally identifiable information except names
- Aggregate data (no geographic identifiers)
- Clear research-only usage statement

---

## Implementation Priority (If Pursuing)

### Phase 1: Easiest to Implement & Test
1. **Indian Railway Stations** - Limited scope, authoritative sources
2. **Wikidata** (small sample) - Structured API, well-documented

### Phase 2: Medium Complexity
3. **Wikipedia Interwiki** - Established APIs, needs filtering
4. **OpenStreetMap** - Requires geographic knowledge, Overpass QL

### Phase 3: High Complexity
5. **Electoral Rolls** - PDF parsing, privacy considerations, state variations

---

## What Would Make These "Validated"?

For each source, implementation should include:

1. **Working Script**
   - Successfully runs without errors
   - Handles API failures gracefully
   - Implements rate limiting

2. **Real Data Output**
   - Generates actual TSV files with transliterations
   - 100+ sample pairs minimum
   - Manual spot-checking shows quality

3. **Documentation**
   - Usage instructions tested by another person
   - Known limitations documented
   - Sample output provided

4. **Quality Metrics**
   - Sample validation (manual check of 50-100 pairs)
   - Error rate measurement
   - Comparison with existing data sources

---

## Alternative: Partner with Existing Projects

Rather than implementing scrapers from scratch, consider:

- **AI4Bharat**: Already has Aksharantar dataset (26M pairs, 21 languages)
- **CVIT-IIIT**: IndicCorp and other Indic NLP datasets
- **Google Dakshina**: Existing 300K pairs (already used)
- **Contribute to existing datasets**: Add new sources to community projects

---

## Next Steps (If Proceeding)

1. **Choose ONE source** to start (recommend Railway Stations - smallest scope)
2. **Build minimal POC** - Test with 10-20 items manually
3. **Validate output** - Manual inspection of quality
4. **Scale gradually** - Expand after validation
5. **Document thoroughly** - Share methodology for reproducibility

---

## Honest Assessment

**These are all untested proposals.** Implementation would require:
- Access to external APIs
- PDF processing infrastructure (for electoral rolls)
- Significant debugging and iteration
- Real-world validation with actual outputs

**Recommendation**: Start with ONE source, validate completely before expanding.

---

## Questions to Answer Before Implementation

1. **Is this data already available elsewhere?**
   - Check AI4Bharat, IndicNLP catalog, ULCA

2. **What's the incremental value?**
   - How much better than existing Dakshina/IIT/affidavits data?

3. **Time investment vs. payoff?**
   - Hours to implement vs. new pairs obtained

4. **Can it be maintained?**
   - APIs change, websites change - sustainability?

---

## Contact & Resources

- Wikidata Query Service: https://query.wikidata.org
- Wikipedia API: https://www.mediawiki.org/wiki/API
- Overpass API: https://overpass-api.de
- Indian Railways: https://indianrailways.gov.in
- State CEO Websites: (varies by state)

---

**Document Status**: Proposal only - no implementation or validation yet.
**Last Updated**: 2024-11-12
**Author**: Claude (AI assistant) for Indicate project
