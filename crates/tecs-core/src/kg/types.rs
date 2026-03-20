use serde::{Deserialize, Serialize};
use std::fmt;

/// 관계 타입 분류 — typed-edge filtering의 핵심
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    /// P31: instance of
    InstanceOf,
    /// P279: subclass of
    SubclassOf,
    /// P361: part of
    PartOf,
    /// P527: has part
    HasPart,
    /// P737: influenced by
    InfluencedBy,
    /// P1542: has effect
    HasEffect,
    /// P2283: uses
    Uses,
    /// P921: main subject
    MainSubject,
    /// 기타 모든 관계
    Other(u32),
}

impl RelationType {
    /// Wikidata property ID에서 RelationType으로 변환
    pub fn from_property_id(pid: &str) -> Self {
        match pid {
            "P31"   => Self::InstanceOf,
            "P279"  => Self::SubclassOf,
            "P361"  => Self::PartOf,
            "P527"  => Self::HasPart,
            "P737"  => Self::InfluencedBy,
            "P1542" => Self::HasEffect,
            "P2283" => Self::Uses,
            "P921"  => Self::MainSubject,
            other => {
                let num = other.trim_start_matches('P')
                    .parse::<u32>()
                    .unwrap_or(0);
                Self::Other(num)
            }
        }
    }

    /// 계층 관계인지 (spanning tree 구축에 사용)
    pub fn is_hierarchical(&self) -> bool {
        matches!(self, Self::InstanceOf | Self::SubclassOf | Self::PartOf | Self::HasPart)
    }

    /// 추론에 높은 가치를 가진 관계인지
    pub fn priority_score(&self) -> u8 {
        match self {
            Self::SubclassOf    => 10,
            Self::InstanceOf    => 9,
            Self::PartOf        => 8,
            Self::HasPart       => 8,
            Self::InfluencedBy  => 7,
            Self::HasEffect     => 7,
            Self::Uses          => 6,
            Self::MainSubject   => 5,
            Self::Other(_)      => 1,
        }
    }
}

/// 지식 그래프 노드
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    /// Wikidata ID (e.g., "Q413")
    pub wikidata_id: String,
    /// 레이블 (e.g., "Physics")
    pub label: String,
    /// Category depth (Sarkar 임베딩용, 나중에 계산)
    pub depth: Option<u32>,
    /// 연결 edge 수 (hub score)
    pub degree: u32,
}

/// 지식 그래프 엣지
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationEdge {
    /// Wikidata property ID 원본
    pub property_id: String,
    /// 분류된 관계 타입
    pub relation_type: RelationType,
    /// 추론 우선순위
    pub priority: u8,
}

impl fmt::Display for EntityNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.label, self.wikidata_id)
    }
}
