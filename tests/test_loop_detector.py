"""Tests for loop detection logic."""

from src.loop_detector import LoopDetector


class TestRepeatDetection:
    def test_no_loop_with_varied_actions(self):
        ld = LoopDetector()
        for action in ["click", "type", "scroll", "click"]:
            result = ld.push(action)
        assert not result.detected

    def test_detects_triple_repeat(self):
        ld = LoopDetector()
        ld.push("click_element")
        ld.push("click_element")
        result = ld.push("click_element")
        assert result.detected
        assert result.pattern_type == "repeat"
        assert "click_element" in result.pattern

    def test_detects_longer_repeat(self):
        ld = LoopDetector()
        results = []
        for _ in range(5):
            results.append(ld.push("scroll_down"))
        # x3 fires (first detection), x4/x5 suppressed
        assert results[2].detected
        assert "x3" in results[2].pattern
        assert not results[3].detected  # suppressed
        assert not results[4].detected  # suppressed

    def test_custom_threshold(self):
        ld = LoopDetector(repeat_threshold=5)
        for _ in range(4):
            result = ld.push("click")
        assert not result.detected
        result = ld.push("click")
        assert result.detected


class TestAlternatingDetection:
    def test_detects_abab_pattern(self):
        ld = LoopDetector()
        ld.push("click")
        ld.push("scroll")
        ld.push("click")
        result = ld.push("scroll")
        assert result.detected
        assert result.pattern_type == "alternating"
        assert "click" in result.pattern
        assert "scroll" in result.pattern

    def test_no_false_positive_ab(self):
        ld = LoopDetector()
        ld.push("click")
        result = ld.push("scroll")
        assert not result.detected

    def test_three_different_no_detection(self):
        ld = LoopDetector()
        for action in ["a", "b", "c", "a", "b", "c"]:
            result = ld.push(action)
        assert not result.detected


class TestWindowAndCounting:
    def test_detected_count_increments(self):
        ld = LoopDetector()
        assert ld.detected_count == 0
        for _ in range(3):
            ld.push("x")
        assert ld.detected_count == 1
        ld.push("x")  # x4 — same pattern suppressed
        assert ld.detected_count == 1

    def test_window_slides(self):
        ld = LoopDetector(window_size=4)
        ld.push("a")
        ld.push("a")
        ld.push("a")  # detected
        ld.push("b")
        ld.push("c")
        ld.push("d")
        result = ld.push("e")
        assert not result.detected

    def test_reset_after_break(self):
        ld = LoopDetector()
        ld.push("x")
        ld.push("x")
        ld.push("x")  # detected
        ld.push("y")  # break
        ld.push("z")
        result = ld.push("w")
        assert not result.detected


class TestSuppression:
    """Tests for loop detection suppression after first detection."""

    def test_suppresses_repeat_after_first_detection(self):
        """x3 fires, x4 and x5 are suppressed."""
        ld = LoopDetector()
        results = []
        for _ in range(5):
            results.append(ld.push("click"))
        # x1, x2 = not detected; x3 = detected; x4, x5 = suppressed
        assert not results[0].detected
        assert not results[1].detected
        assert results[2].detected
        assert not results[3].detected
        assert not results[4].detected
        assert ld.detected_count == 1

    def test_suppression_resets_when_pattern_breaks(self):
        """After break, new loop fires again."""
        ld = LoopDetector()
        for _ in range(4):
            ld.push("click")
        assert ld.detected_count == 1
        # Break the pattern
        ld.push("scroll")
        # New loop — need 3 consecutive "type" (the break resets suppression)
        detected_any = False
        for _ in range(3):
            result = ld.push("type")
            if result.detected:
                detected_any = True
        assert detected_any
        assert ld.detected_count == 2

    def test_alternating_suppressed_after_first(self):
        """ABAB fires once, further ABAB pushes are suppressed."""
        ld = LoopDetector()
        ld.push("click")
        ld.push("scroll")
        ld.push("click")
        r4 = ld.push("scroll")  # ABAB detected
        assert r4.detected
        assert ld.detected_count == 1
        # Continue alternating — suppressed
        r5 = ld.push("click")
        r6 = ld.push("scroll")
        assert not r5.detected
        assert not r6.detected
        assert ld.detected_count == 1

    def test_different_elements_no_false_positive(self):
        """Clicking different elements should not trigger: click[5], click[12], click[7]."""
        ld = LoopDetector()
        actions = ["click_element[5]", "click_element[12]", "click_element[7]"]
        for action in actions:
            result = ld.push(action)
        assert not result.detected

    def test_same_element_triggers_detection(self):
        """Clicking the same element 3x should trigger: click[5], click[5], click[5]."""
        ld = LoopDetector()
        for _ in range(3):
            result = ld.push("click_element[5]")
        assert result.detected
        assert "click_element[5]" in result.pattern


class TestQualifyActionName:
    """Tests for the _qualify_action_name helper in orchestrator."""

    def test_click_with_index(self):
        from src.orchestrator import _qualify_action_name
        assert _qualify_action_name("click_element", {"index": 5}) == "click_element[5]"

    def test_input_with_index(self):
        from src.orchestrator import _qualify_action_name
        assert _qualify_action_name("input_text", {"index": 3, "text": "hello"}) == "input_text[3]"

    def test_scroll_down(self):
        from src.orchestrator import _qualify_action_name
        assert _qualify_action_name("scroll", {"down": True}) == "scroll_down"

    def test_scroll_up(self):
        from src.orchestrator import _qualify_action_name
        assert _qualify_action_name("scroll", {"down": False}) == "scroll_up"

    def test_scroll_empty_dict_returns_bare(self):
        """Empty dict is falsy — returns bare action key."""
        from src.orchestrator import _qualify_action_name
        assert _qualify_action_name("scroll", {}) == "scroll"

    def test_no_params_returns_bare(self):
        from src.orchestrator import _qualify_action_name
        assert _qualify_action_name("done", None) == "done"

    def test_empty_dict_returns_bare(self):
        from src.orchestrator import _qualify_action_name
        assert _qualify_action_name("go_to_url", {}) == "go_to_url"

    def test_go_to_url_no_index(self):
        from src.orchestrator import _qualify_action_name
        assert _qualify_action_name("go_to_url", {"url": "https://example.com"}) == "go_to_url"
