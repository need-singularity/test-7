use dashmap::DashMap;
use nalgebra::DMatrix;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// 거리 매트릭스 + PH 결과 캐시
pub struct ResultCache {
    /// distance matrix 캐시
    dm_cache: DashMap<u64, DMatrix<f64>>,
    /// 최대 캐시 엔트리 수
    max_entries: usize,
}

impl ResultCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            dm_cache: DashMap::with_capacity(max_entries),
            max_entries,
        }
    }

    /// 노드 목록으로부터 캐시 키 생성
    pub fn cache_key(node_ids: &mut [u32]) -> u64 {
        node_ids.sort();
        let mut hasher = DefaultHasher::new();
        node_ids.hash(&mut hasher);
        hasher.finish()
    }

    /// distance matrix 캐시 조회
    pub fn get_dm(&self, key: u64) -> Option<DMatrix<f64>> {
        self.dm_cache.get(&key).map(|v| v.clone())
    }

    /// distance matrix 캐시 저장
    pub fn put_dm(&self, key: u64, dm: DMatrix<f64>) {
        if self.dm_cache.len() >= self.max_entries {
            // 간단한 eviction: 가장 오래된 것 제거 (실제로는 LRU 필요)
            if let Some(entry) = self.dm_cache.iter().next() {
                let key_to_remove = *entry.key();
                drop(entry);
                self.dm_cache.remove(&key_to_remove);
            }
        }
        self.dm_cache.insert(key, dm);
    }

    /// 캐시 비우기
    pub fn clear(&self) {
        self.dm_cache.clear();
    }

    pub fn len(&self) -> usize {
        self.dm_cache.len()
    }
}
