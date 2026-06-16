"""Test suite for raw data generation and validation — 5 scenarios."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


class TestRawDataTeachers:
    """Scenarios 1-5: Validate teacher raw data."""

    def test_teacher_record_completeness(self, raw_teachers: List[Dict[str, Any]]):
        """Scenario 1: Every teacher record has all required fields."""
        required = {"teacher_id", "name", "gender", "department", "designation", "qualification", "experience_years", "subjects", "email"}
        for t in raw_teachers:
            assert required.issubset(t.keys()), f"Teacher {t.get('teacher_id')} missing fields"

    def test_teacher_gender_diversity(self, raw_teachers: List[Dict[str, Any]]):
        """Scenario 2: Teacher data includes both male and female teachers."""
        genders = {t["gender"] for t in raw_teachers}
        assert "Male" in genders
        assert "Female" in genders
        # Verify specific female teachers exist
        assert any(t["name"] == "Anita Sharma" for t in raw_teachers)
        assert any(t["name"] == "Priya Verma" for t in raw_teachers)

    def test_teacher_department_diversity(self, raw_teachers: List[Dict[str, Any]]):
        """Scenario 3: Teachers span multiple departments."""
        depts = {t["department"] for t in raw_teachers}
        assert len(depts) >= 3, f"Expected ≥3 departments, got {depts}"
        assert "Computer Science" in depts

    def test_teacher_experience_range(self, raw_teachers: List[Dict[str, Any]]):
        """Scenario 4: Teachers have varied experience levels."""
        years = [t["experience_years"] for t in raw_teachers]
        assert min(years) >= 0
        assert max(years) >= 6
        assert len(set(years)) >= 3

    def test_teacher_subjects_unique(self, raw_teachers: List[Dict[str, Any]]):
        """Scenario 5: Teacher subjects lists are non-empty and varied."""
        all_subjects = []
        for t in raw_teachers:
            assert len(t["subjects"]) > 0, f"Teacher {t['teacher_id']} has no subjects"
            all_subjects.extend(t["subjects"])
        assert len(set(all_subjects)) >= 8, "Need more subject variety"


class TestRawDataStudents:
    """Scenarios 6-10: Validate student raw data."""

    def test_student_record_completeness(self, raw_students: List[Dict[str, Any]]):
        """Scenario 6: Every student record has all required fields."""
        required = {"student_id", "name", "gender", "department", "semester", "roll_number", "email", "batch", "cgpa"}
        for s in raw_students:
            assert required.issubset(s.keys()), f"Student {s.get('student_id')} missing fields"

    def test_student_count_minimum(self, raw_students: List[Dict[str, Any]]):
        """Scenario 7: At least 10 student records exist."""
        assert len(raw_students) >= 10

    def test_student_gender_diversity(self, raw_students: List[Dict[str, Any]]):
        """Scenario 8: Students include both genders."""
        genders = {s["gender"] for s in raw_students}
        assert "Male" in genders
        assert "Female" in genders

    def test_student_semester_range(self, raw_students: List[Dict[str, Any]]):
        """Scenario 9: Students are distributed across semesters (2, 4, 6, 8)."""
        semesters = {s["semester"] for s in raw_students}
        assert len(semesters) >= 3, f"Expected ≥3 different semesters, got {semesters}"

    def test_student_cgpa_range(self, raw_students: List[Dict[str, Any]]):
        """Scenario 10: Student CGPA covers a realistic range."""
        cgpas = [s["cgpa"] for s in raw_students]
        assert all(0 <= c <= 10 for c in cgpas)
        assert max(cgpas) - min(cgpas) >= 1.5


class TestRawDataCourses:
    """Scenarios 11-14: Validate course raw data."""

    def test_course_record_completeness(self, raw_courses: List[Dict[str, Any]]):
        """Scenario 11: Every course record has all required fields."""
        required = {"course_code", "course_name", "department", "credits", "instructor_id", "semester", "syllabus_summary"}
        for c in raw_courses:
            assert required.issubset(c.keys()), f"Course {c.get('course_code')} missing fields"

    def test_course_multiple_departments(self, raw_courses: List[Dict[str, Any]]):
        """Scenario 12: Courses span at least 3 departments."""
        depts = {c["department"] for c in raw_courses}
        assert len(depts) >= 3

    def test_course_credit_range(self, raw_courses: List[Dict[str, Any]]):
        """Scenario 13: Credits are 3 or 4."""
        for c in raw_courses:
            assert c["credits"] in (3, 4), f"Course {c['course_code']} has invalid credits {c['credits']}"

    def test_course_syllabus_nonempty(self, raw_courses: List[Dict[str, Any]]):
        """Scenario 14: Each course has a non-empty syllabus summary."""
        for c in raw_courses:
            assert len(c["syllabus_summary"]) > 20


class TestRawDataDepartments:
    """Scenarios 15-17: Validate department raw data."""

    def test_department_record_completeness(self, raw_departments: List[Dict[str, Any]]):
        """Scenario 15: Each department record has all required fields."""
        required = {"dept_code", "dept_name", "hod_name", "total_faculty", "total_students"}
        for d in raw_departments:
            assert required.issubset(d.keys())

    def test_department_count_minimum(self, raw_departments: List[Dict[str, Any]]):
        """Scenario 16: At least 4 departments."""
        assert len(raw_departments) >= 4

    def test_department_student_faculty_ratio(self, raw_departments: List[Dict[str, Any]]):
        """Scenario 17: Each department has realistic student-to-faculty ratios."""
        for d in raw_departments:
            ratio = d["total_students"] / d["total_faculty"]
            assert 5 <= ratio <= 30, f"Department {d['dept_code']} ratio {ratio:.1f} seems off"


class TestRawDataAggregate:
    """Scenarios 18-20: Aggregate data validation."""

    def test_total_records_count(self, all_raw_data: dict):
        """Scenario 18: Total raw data records is at least 25."""
        assert all_raw_data["total_count"] >= 25

    def test_teacher_student_ratio(self, raw_teachers, raw_students):
        """Scenario 19: Teacher-to-student ratio is realistic."""
        ratio = len(raw_students) / len(raw_teachers)
        assert 1.5 <= ratio <= 5.0

    def test_cross_reference_instructors(self, raw_teachers, raw_courses):
        """Scenario 20: Course instructor_ids reference valid teacher_ids."""
        teacher_ids = {t["teacher_id"] for t in raw_teachers}
        for c in raw_courses:
            assert c["instructor_id"] in teacher_ids, f"Course {c['course_code']} references unknown instructor {c['instructor_id']}"

    def test_course_code_format(self, raw_courses):
        """Scenario 20b: Course codes follow XX123 format."""
        import re
        for c in raw_courses:
            assert re.match(r"^[A-Z]{2}\d{3}$", c["course_code"]), f"Bad course code: {c['course_code']}"

    def test_student_id_format(self, raw_students):
        """Scenario 20c: Student IDs follow SXXX format and roll numbers match department."""
        import re
        for s in raw_students:
            assert re.match(r"^S\d{3}$", s["student_id"]), f"Bad student ID: {s['student_id']}"
            dept_prefix = {"Computer Science": "CS", "Electronics": "EC", "Mathematics": "MA", "Mechanical Engineering": "ME"}
            expected_prefix = dept_prefix.get(s["department"])
            if expected_prefix:
                assert s["roll_number"].startswith(expected_prefix), f"Roll {s['roll_number']} doesn't match dept {s['department']}"
