"""Which records are one paper, and what the merged paper says (FR-D3, D145-D148).

What fails silently here is a merge that hides a paper (the user sees one entry and
can add only one), a missed merge that shows a paper twice, and a merged entry that
says something no member said. Every case below is a shape the D146 fixture found in
real records; the fixture itself is scored in `test_eval_dedup.py`.
"""

from __future__ import annotations

from datetime import date

import pytest

from scholiast.domain.dedup import (
    SHIPPED_RULE,
    DiscoveredWork,
    MatchReason,
    MatchRule,
    group,
    merge,
    name_words,
    same_work,
    works,
)
from scholiast.domain.discovery import LinkKind, OpenAccessLink, PaperCandidate
from scholiast.domain.errors import InvariantViolation

TITLE = "Dense Passage Retrieval for Open-Domain Question Answering"


def record(source: str = "openalex", n: int = 1, **fields: object) -> PaperCandidate:
    base: dict[str, object] = {
        "source": source,
        "source_id": f"{source}-{n}",
        "rank": n,
        "title": TITLE,
        "authors": ("Vladimir Karpukhin", "Barlas Oguz"),
        "year": 2020,
        "adapter_version": "test-1",
    }
    base.update(fields)
    return PaperCandidate(**base)  # type: ignore[arg-type]


def pdf(url: str) -> OpenAccessLink:
    return OpenAccessLink(url, LinkKind.PDF)


class TestAnIdentifierDecides:
    def test_a_shared_arxiv_id_merges_whatever_the_titles_say(self) -> None:
        # D146: OpenAlex records carrying a paper's arXiv id under an unrelated
        # title; the owner labelled them the same work.
        corrupted = record(
            "openalex", title="LoRA Fine-Tuning of a 3B Code LLM", arxiv_id="2106.09685"
        )
        lora = record("arxiv", title="LoRA: Low-Rank Adaptation", arxiv_id="2106.09685")

        assert same_work(corrupted, lora) is MatchReason.ARXIV_ID

    def test_a_shared_publisher_doi_merges_whatever_the_titles_say(self) -> None:
        a = record(
            "openalex", 1, title="Geometry-enhanced molecular representation", doi="10.1038/x"
        )
        b = record("openalex", 2, title="ChemRL-GEM: something else entirely", doi="10.1038/x")

        assert same_work(a, b) is MatchReason.DOI

    def test_arxivs_datacite_doi_is_the_arxiv_id_it_spells(self) -> None:
        # OpenAlex keeps 10.48550/arxiv.X as the DOI; arXiv reports the bare id (D132).
        openalex = record("openalex", doi="10.48550/arxiv.2004.04906", title="Different")
        arxiv = record("arxiv", arxiv_id="2004.04906")

        assert same_work(openalex, arxiv) is MatchReason.ARXIV_ID

    def test_two_arxiv_ids_are_two_papers_even_with_one_title(self) -> None:
        # "A Survey of ..." is a title more than one submission has had (D148).
        a = record("arxiv", 1, title="A Survey of Large Language Models", arxiv_id="2303.18223")
        b = record("arxiv", 2, title="A Survey of Large Language Models", arxiv_id="2402.06196")

        assert same_work(a, b) is None


class TestATitleIsEvidence:
    def test_a_preprint_and_its_camera_ready_are_one_work(self) -> None:
        preprint = record("arxiv", arxiv_id="2004.04906")
        published = record("openalex", doi="10.18653/v1/2020.emnlp-main.550")

        assert same_work(preprint, published) is MatchReason.TITLE

    def test_punctuation_and_case_do_not_separate_titles(self) -> None:
        a = record(title="Exploring Chain-of-Thought Style Prompting for Text-to-SQL")
        b = record(title="Exploring chain of thought style prompting for text to SQL", n=2)

        assert same_work(a, b) is MatchReason.TITLE

    def test_a_title_overlap_exactly_at_the_threshold_is_a_match(self) -> None:
        # D148 says "at least T". Four shared words of five: Jaccard 0.8.
        rule = MatchRule(title_threshold=0.8, year_tolerance=1)
        a = record(title="alpha beta gamma delta epsilon")
        b = record(title="alpha beta gamma delta", n=2)

        assert same_work(a, b, rule) is MatchReason.TITLE

    def test_titles_below_the_threshold_are_two_papers(self) -> None:
        a = record(title="Dense Passage Retrieval for Open-Domain Question Answering")
        b = record(title="Dense Hierarchical Retrieval for Open-Domain Question Answering", n=2)

        assert same_work(a, b) is None

    def test_a_journal_version_two_years_on_is_one_work(self) -> None:
        # LRCN: a 2014 technical report and its 2016 TPAMI extension (D146).
        assert same_work(record(year=2014), record(year=2016, n=2)) is MatchReason.TITLE

    def test_years_further_apart_than_the_tolerance_are_two_papers(self) -> None:
        late = 2020 + SHIPPED_RULE.year_tolerance + 1
        assert same_work(record(year=2020), record(year=late, n=2)) is None

    def test_an_unknown_year_is_no_evidence_against(self) -> None:
        assert same_work(record(year=None), record(year=1990, n=2)) is MatchReason.TITLE

    def test_one_title_by_different_first_authors_is_two_papers(self) -> None:
        a = record(authors=("Wayne Xin Zhao",), title="A Survey of Large Language Models")
        b = record(authors=("Shervin Minaee",), title="A Survey of Large Language Models", n=2)

        assert same_work(a, b) is None

    def test_unknown_authors_are_no_evidence_against(self) -> None:
        assert same_work(record(authors=()), record(n=2)) is MatchReason.TITLE

    def test_a_first_author_known_only_by_initials_is_unknown_not_a_mismatch(self) -> None:
        # Initials identify nobody, so "J. K." can neither confirm nor refute.
        assert same_work(record(authors=("J. K.",)), record(n=2)) is MatchReason.TITLE

    def test_a_first_author_spelled_differently_but_sharing_a_name_still_matches(self) -> None:
        # D146: "Zequan Liu" (arXiv) against "Zhaojia Liu" (OpenAlex), one paper.
        a = record(authors=("Zequan Liu",))
        b = record(authors=("Zhaojia Liu",), n=2)

        assert same_work(a, b) is MatchReason.TITLE


class TestNameWords:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Ho, Jonathan", "Jonathan Ho"),
            ("Ge Zheng", "Zheng Ge"),
            ("Kerim Büyükakyüz", "Kerim Buyukakyuz"),
        ],
    )
    def test_one_person_spelled_two_ways_is_one_set_of_names(self, a: str, b: str) -> None:
        assert name_words(a) == name_words(b)

    def test_an_initial_identifies_nobody(self) -> None:
        # Otherwise "J. Zhang" and "J. Smith" would share an author.
        assert name_words("J. Edward Hu") == frozenset({"edward", "hu"})


class TestGrouping:
    def test_merges_are_transitive(self) -> None:
        # The preprint matches the camera-ready by title, the camera-ready matches a
        # second OpenAlex record by DOI: one work of three, though the preprint and
        # the second record match nothing directly. The camera-ready comes last, so
        # it is joined to two works that already exist, not to two lone records.
        preprint = record("arxiv", 1, arxiv_id="2004.04906")
        duplicate = record("openalex", 2, doi="10.1/a", title="Unrelated title")
        camera_ready = record("openalex", 1, doi="10.1/a")

        assert group([preprint, duplicate, camera_ready]) == ((0, 1, 2),)

    def test_a_work_sits_where_its_earliest_record_sat(self) -> None:
        other = record("openalex", 1, title="Something else", authors=("Someone Else",))
        late = record("openalex", 2)
        early = record("arxiv", 1)

        found = works([early, other, late])

        assert [w.members for w in found] == [(early, late), (other,)]


class TestMerging:
    def test_the_merged_paper_takes_each_field_from_the_member_that_has_it(self) -> None:
        preprint = record(
            "arxiv",
            year=2020,
            arxiv_id="2004.04906",
            abstract="Short.",
            oa_links=(pdf("https://arxiv.org/pdf/2004.04906v3"),),
        )
        published = record(
            "openalex",
            year=2021,
            doi="10.18653/v1/2020.emnlp-main.550",
            citation_count=4100,
            abstract="A longer abstract from the published version.",
            oa_links=(
                pdf("https://arxiv.org/pdf/2004.04906v3"),
                pdf("https://aclanthology.org/x.pdf"),
            ),
        )

        work = merge([preprint, published])

        assert work.doi == "10.18653/v1/2020.emnlp-main.550"
        assert work.arxiv_id == "2004.04906"
        assert work.year == 2020, "the work first appeared with its preprint (FR-D4)"
        assert work.abstract == "A longer abstract from the published version."
        assert work.citation_count == 4100
        assert [link.url for link in work.oa_links] == [
            "https://arxiv.org/pdf/2004.04906v3",
            "https://aclanthology.org/x.pdf",
        ], "every link once, in member order"

    def test_one_work_held_twice_by_openalex_does_not_count_its_citations_twice(self) -> None:
        a = record("openalex", 1, citation_count=30)
        b = record("openalex", 2, citation_count=12)

        assert merge([a, b]).citation_count == 30

    def test_no_member_reporting_citations_is_unknown_not_zero(self) -> None:
        assert merge([record("arxiv"), record("arxiv", 2)]).citation_count is None

    def test_an_arxiv_datacite_doi_is_not_offered_as_the_publisher_doi(self) -> None:
        work = merge([record("openalex", doi="10.48550/arxiv.2004.04906")])

        assert (work.doi, work.arxiv_id) == (None, "2004.04906")


class TestTheRecordToAdd:
    def test_prefers_an_arxiv_record_the_fetcher_can_follow(self) -> None:
        fetchable_openalex = record("openalex", oa_links=(pdf("https://aclanthology.org/x.pdf"),))
        fetchable_arxiv = record(
            "arxiv", arxiv_id="2004.04906", oa_links=(pdf("https://arxiv.org/pdf/2004.04906v3"),)
        )

        work = merge([fetchable_openalex, fetchable_arxiv], fetchable=lambda r: bool(r.oa_links))

        assert work.add is fetchable_arxiv

    def test_an_arxiv_id_the_fetcher_cannot_follow_loses_to_a_fetchable_record(self) -> None:
        unreachable_arxiv = record("arxiv", arxiv_id="2004.04906")
        reachable = record("openalex", oa_links=(pdf("https://aclanthology.org/x.pdf"),))

        work = merge([unreachable_arxiv, reachable], fetchable=lambda r: bool(r.oa_links))

        assert work.add is reachable

    def test_with_nothing_fetchable_the_first_record_is_offered(self) -> None:
        first, second = record("openalex"), record("arxiv", arxiv_id="2004.04906")

        assert merge([first, second]).add is first

    def test_what_is_shown_is_what_will_be_added(self) -> None:
        shown = record(
            "arxiv",
            title="DDCoT: Duty-Distinct Chain-of-Thought Prompting",
            authors=("Ge Zheng",),
            venue=None,
            published=date(2023, 10, 25),
            arxiv_id="2310.16436",
        )
        other = record(
            "openalex", title="DDCoT", authors=("Zheng Ge",), venue="NeurIPS", arxiv_id="2310.16436"
        )

        work = merge([other, shown], fetchable=lambda r: r is shown)

        assert (work.title, work.authors, work.venue, work.published) == (
            shown.title,
            shown.authors,
            shown.venue,
            shown.published,
        )


class TestInvariants:
    def test_a_work_without_records_cannot_exist(self) -> None:
        with pytest.raises(InvariantViolation):
            DiscoveredWork((), record(), None, None, None, None, None, ())

    def test_the_record_to_add_must_be_one_of_the_works_records(self) -> None:
        with pytest.raises(InvariantViolation):
            DiscoveredWork((record(),), record("arxiv"), None, None, None, None, None, ())

    @pytest.mark.parametrize(("threshold", "tolerance"), [(0.0, 1), (1.01, 1), (0.8, -1)])
    def test_a_rule_outside_its_range_is_refused(self, threshold: float, tolerance: int) -> None:
        with pytest.raises(InvariantViolation):
            MatchRule(threshold, tolerance)
