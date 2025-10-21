package com.drishtikon.stocksync.cache;

import com.drishtikon.stocksync.dao.ProductDAO;
import com.drishtikon.stocksync.model.Product;
import com.drishtikon.stocksync.util.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Day 6: Write-through cache for Product entities.
 * 
 * Uses ConcurrentHashMap for thread-safe caching.
 * Cache strategy: Write-through (DB first, then cache).
 * DB is always the source of truth.
 */
public class CachedProductDAO {
    private static final Logger logger = LoggerFactory.getLogger(CachedProductDAO.class);
    
    private final ProductDAO productDAO;
    private final ConcurrentHashMap<Long, Product> cacheById;
    private final ConcurrentHashMap<String, Product> cacheBySku;
    
    // Cache statistics
    private final AtomicLong cacheHits = new AtomicLong(0);
    private final AtomicLong cacheMisses = new AtomicLong(0);

    public CachedProductDAO(ProductDAO productDAO) {
        this.productDAO = productDAO;
        this.cacheById = new ConcurrentHashMap<>();
        this.cacheBySku = new ConcurrentHashMap<>();
    }

    /**
     * Warm up cache on startup by loading all active products
     */
    public void warmUpCache() {
        logger.info("Warming up cache...");
        Result<List<Product>> result = productDAO.findAll();
        
        if (result.isSuccess()) {
            List<Product> products = result.getOrThrow();
            products.forEach(this::updateCache);
            logger.info("Cache warmed up with {} products", products.size());
        } else {
            logger.error("Failed to warm up cache: {}", result.getError());
        }
    }

    /**
     * Find by ID with cache lookup
     */
    public Result<Optional<Product>> findById(Long id) {
        if (id == null || id <= 0) {
            return Result.failure("Invalid product ID");
        }

        // Check cache first
        Product cached = cacheById.get(id);
        if (cached != null && !cached.isDeleted()) {
            cacheHits.incrementAndGet();
            logger.debug("Cache hit for product id={}", id);
            return Result.success(Optional.of(cached));
        }

        // Cache miss - fetch from DB
        cacheMisses.incrementAndGet();
        logger.debug("Cache miss for product id={}", id);
        
        Result<Optional<Product>> result = productDAO.findById(id);
        if (result.isSuccess()) {
            result.getValue().ifPresent(opt -> opt.ifPresent(this::updateCache));
        }
        
        return result;
    }

    /**
     * Find by SKU with cache lookup
     */
    public Result<Optional<Product>> findBySku(String sku) {
        if (sku == null || sku.isBlank()) {
            return Result.failure("Invalid SKU");
        }

        // Check cache first
        Product cached = cacheBySku.get(sku);
        if (cached != null && !cached.isDeleted()) {
            cacheHits.incrementAndGet();
            logger.debug("Cache hit for product sku={}", sku);
            return Result.success(Optional.of(cached));
        }

        // Cache miss - fetch from DB
        cacheMisses.incrementAndGet();
        logger.debug("Cache miss for product sku={}", sku);
        
        Result<Optional<Product>> result = productDAO.findBySku(sku);
        if (result.isSuccess()) {
            result.getValue().ifPresent(opt -> opt.ifPresent(this::updateCache));
        }
        
        return result;
    }

    /**
     * Find all products (no caching for list operations)
     */
    public Result<List<Product>> findAll() {
        return productDAO.findAll();
    }

    /**
     * Save product with write-through cache
     * DB is updated first, then cache
     */
    public Result<Product> save(Product product) {
        if (product == null) {
            return Result.failure("Product cannot be null");
        }

        // Write to DB first
        Result<Product> result = productDAO.save(product);
        
        if (result.isSuccess()) {
            Product savedProduct = result.getOrThrow();
            // Update cache after successful DB write
            updateCache(savedProduct);
            logger.debug("Cache updated for product id={}, sku={}", 
                savedProduct.getId(), savedProduct.getSku());
        }
        
        return result;
    }

    /**
     * Delete product with cache invalidation
     */
    public Result<Void> delete(Long id) {
        if (id == null || id <= 0) {
            return Result.failure("Invalid product ID");
        }

        // Delete from DB first
        Result<Void> result = productDAO.delete(id);
        
        if (result.isSuccess()) {
            // Invalidate cache after successful DB delete
            Product cached = cacheById.remove(id);
            if (cached != null) {
                cacheBySku.remove(cached.getSku());
                logger.debug("Cache invalidated for product id={}, sku={}", 
                    id, cached.getSku());
            }
        }
        
        return result;
    }

    /**
     * Batch insert with cache update
     */
    public Result<Integer> batchInsert(List<Product> products) {
        if (products == null || products.isEmpty()) {
            return Result.failure("Product list cannot be empty");
        }

        Result<Integer> result = productDAO.batchInsert(products);
        
        if (result.isSuccess()) {
            // Note: Batch insert doesn't return IDs, so we can't cache them
            // Cache will be populated on next read
            logger.info("Batch insert completed, cache will be updated on next read");
        }
        
        return result;
    }

    /**
     * Update cache with product
     */
    private void updateCache(Product product) {
        if (product != null && !product.isDeleted()) {
            cacheById.put(product.getId(), product);
            cacheBySku.put(product.getSku(), product);
        }
    }

    /**
     * Clear all cache
     */
    public void clearCache() {
        cacheById.clear();
        cacheBySku.clear();
        cacheHits.set(0);
        cacheMisses.set(0);
        logger.info("Cache cleared");
    }

    /**
     * Get cache statistics
     */
    public CacheStats getCacheStats() {
        long hits = cacheHits.get();
        long misses = cacheMisses.get();
        long total = hits + misses;
        double hitRate = total > 0 ? (double) hits / total * 100 : 0;
        
        return new CacheStats(hits, misses, hitRate, cacheById.size());
    }

    public static class CacheStats {
        private final long hits;
        private final long misses;
        private final double hitRate;
        private final int size;

        public CacheStats(long hits, long misses, double hitRate, int size) {
            this.hits = hits;
            this.misses = misses;
            this.hitRate = hitRate;
            this.size = size;
        }

        public long getHits() {
            return hits;
        }

        public long getMisses() {
            return misses;
        }

        public double getHitRate() {
            return hitRate;
        }

        public int getSize() {
            return size;
        }

        @Override
        public String toString() {
            return String.format("CacheStats{hits=%d, misses=%d, hitRate=%.2f%%, size=%d}", 
                hits, misses, hitRate, size);
        }
    }
}
