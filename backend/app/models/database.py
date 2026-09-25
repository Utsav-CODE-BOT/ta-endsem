from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from backend.app.database.session import Base


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, index=True)
    source = Column(String(128), default="General Feedback", index=True)
    original_text = Column(Text, nullable=False)
    clean_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=True, index=True)
    language = Column(String(16), default="en")
    word_count = Column(Integer, default=0)
    meta_info = Column(Text, default="{}")  # JSON encoded metadata

    # Relationships
    nlp_result = relationship("NLPResultModel", back_populates="document", uselist=False, cascade="all, delete-orphan")
    memberships = relationship("IssueMembershipModel", back_populates="document", cascade="all, delete-orphan")


class NLPResultModel(Base):
    __tablename__ = "nlp_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    sentiment = Column(String(32), index=True)
    sentiment_score = Column(Float, default=0.0)
    emotion = Column(String(32), index=True)
    emotion_score = Column(Float, default=0.0)
    aspects = Column(Text, default="[]")  # JSON encoded list of aspect objects

    document = relationship("DocumentModel", back_populates="nlp_result")


class IssueModel(Base):
    __tablename__ = "issues"

    issue_id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, default="")
    cluster_size = Column(Integer, default=0)
    percentage = Column(Float, default=0.0)
    top_terms = Column(Text, default="[]")  # JSON encoded list of keywords
    center_2d = Column(Text, default="[0.0, 0.0]")  # JSON encoded [x, y]
    created_at = Column(DateTime, default=datetime.utcnow)

    memberships = relationship("IssueMembershipModel", back_populates="issue", cascade="all, delete-orphan")
    metrics = relationship("IssueMetricModel", back_populates="issue", cascade="all, delete-orphan")
    hypotheses = relationship("RootCauseHypothesisModel", back_populates="issue", cascade="all, delete-orphan")


class IssueMembershipModel(Base):
    __tablename__ = "issue_memberships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    issue_id = Column(Integer, ForeignKey("issues.issue_id", ondelete="CASCADE"), index=True)
    similarity = Column(Float, default=0.0)
    coord_x = Column(Float, default=0.0)
    coord_y = Column(Float, default=0.0)

    document = relationship("DocumentModel", back_populates="memberships")
    issue = relationship("IssueModel", back_populates="memberships")


class IssueMetricModel(Base):
    __tablename__ = "issue_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    issue_id = Column(Integer, ForeignKey("issues.issue_id", ondelete="CASCADE"), index=True)
    emerging_score = Column(Float, default=0.0, index=True)
    priority = Column(String(32), default="MEDIUM", index=True)
    growth_rate = Column(Float, default=0.0)
    recent_volume = Column(Integer, default=0)
    negative_ratio = Column(Float, default=0.0)
    coherence = Column(Float, default=0.75)
    first_detected_date = Column(String(64), default="")
    history_json = Column(Text, default="[]")  # Time series history points

    issue = relationship("IssueModel", back_populates="metrics")


class RootCauseHypothesisModel(Base):
    __tablename__ = "root_cause_hypotheses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    issue_id = Column(Integer, ForeignKey("issues.issue_id", ondelete="CASCADE"), index=True)
    candidate_factor = Column(String(256), nullable=False)
    factor_type = Column(String(64), default="Associated Factor")
    confidence = Column(Float, default=0.0)
    lift = Column(Float, default=1.0)
    pmi = Column(Float, default=0.0)
    temporal_relationship = Column(String(64), default="Moderate")
    co_occurrence_count = Column(Integer, default=0)
    total_mention_count = Column(Integer, default=0)
    interpretation = Column(Text, default="")

    issue = relationship("IssueModel", back_populates="hypotheses")
