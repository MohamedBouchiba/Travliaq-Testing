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
        for _ in range(5):
            result = ld.push("scroll_down")
        assert result.detected
        assert "x5" in result.pattern

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
        ld.push("x")
        assert ld.detected_count == 2

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
