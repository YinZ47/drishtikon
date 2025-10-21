package com.drishtikon.stocksync;

import com.drishtikon.stocksync.cache.CachedProductDAO;
import com.drishtikon.stocksync.dao.ProductDAO;
import com.drishtikon.stocksync.dao.SchemaInitializer;
import com.drishtikon.stocksync.health.DatabaseHealthCheck;
import com.drishtikon.stocksync.model.Product;
import com.drishtikon.stocksync.util.ConnectionPoolFactory;
import com.drishtikon.stocksync.util.DatabaseBackup;
import com.drishtikon.stocksync.util.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.math.BigDecimal;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * StockSync Main Application
 * 
 * Demonstrates all components from the 10-Day Database Sprint:
 * - Day 1-2: Product with Builder pattern, Result wrapper
 * - Day 3: SQLite schema with triggers and indexes
 * - Day 4: JDBC CRUD operations
 * - Day 5: HikariCP connection pooling and transactions
 * - Day 6: Write-through caching
 * - Day 7-8: Performance optimization and batch operations
 * - Day 9: Logging and error handling
 * - Day 10: Health checks and backup/restore
 */
public class StockSyncApp {
    private static final Logger logger = LoggerFactory.getLogger(StockSyncApp.class);
    
    private final DataSource dataSource;
    private final ProductDAO productDAO;
    private final CachedProductDAO cachedProductDAO;
    private final DatabaseHealthCheck healthCheck;

    public StockSyncApp(String dbPath) throws SQLException {
        logger.info("Initializing StockSync application with database: {}", dbPath);
        
        // Day 5: Create connection pool
        this.dataSource = ConnectionPoolFactory.createDataSource("jdbc:sqlite:" + dbPath);
        
        // Day 3: Initialize database schema
        try (Connection conn = dataSource.getConnection()) {
            SchemaInitializer.initializeSchema(conn);
        }
        
        // Day 4: Create DAO
        this.productDAO = new ProductDAO(dataSource);
        
        // Day 6: Create cached DAO and warm up cache
        this.cachedProductDAO = new CachedProductDAO(productDAO);
        this.cachedProductDAO.warmUpCache();
        
        // Day 10: Create health check
        this.healthCheck = new DatabaseHealthCheck(dataSource);
        
        logger.info("StockSync application initialized successfully");
    }

    /**
     * Run demonstration of all features
     */
    public void runDemo() {
        logger.info("=== Starting StockSync Demo ===");
        
        // Day 10: Health check
        performHealthCheck();
        
        // Day 1-2: Create products using Builder pattern
        demonstrateProductCreation();
        
        // Day 4: CRUD operations
        demonstrateCRUDOperations();
        
        // Day 6: Cache performance
        demonstrateCachePerformance();
        
        // Day 8: Batch operations
        demonstrateBatchOperations();
        
        // Day 5: Transaction management
        demonstrateTransactions();
        
        // Day 10: Backup
        demonstrateBackup();
        
        logger.info("=== Demo Completed ===");
    }

    private void performHealthCheck() {
        logger.info("--- Health Check ---");
        DatabaseHealthCheck.HealthStatus status = healthCheck.getHealthStatus();
        logger.info("Health Status: {}", status);
    }

    private void demonstrateProductCreation() {
        logger.info("--- Product Creation with Builder Pattern ---");
        
        // Create product using builder
        Product widget = Product.builder()
            .sku("WDG-001")
            .name("Super Widget")
            .quantity(100)
            .price(new BigDecimal("29.99"))
            .build();
        
        logger.info("Created product: {}", widget);
        
        // Save product
        Result<Product> result = cachedProductDAO.save(widget);
        if (result.isSuccess()) {
            logger.info("Product saved successfully: {}", result.getOrThrow());
        } else {
            logger.error("Failed to save product: {}", result.getError());
        }
    }

    private void demonstrateCRUDOperations() {
        logger.info("--- CRUD Operations ---");
        
        // Create
        Product gadget = Product.builder()
            .sku("GDG-001")
            .name("Amazing Gadget")
            .quantity(50)
            .price(new BigDecimal("49.99"))
            .build();
        
        Result<Product> createResult = cachedProductDAO.save(gadget);
        if (createResult.isFailure()) {
            logger.error("Create failed: {}", createResult.getError());
            return;
        }
        
        Product savedGadget = createResult.getOrThrow();
        logger.info("Created: {}", savedGadget);
        
        // Read by ID
        Result<Optional<Product>> readResult = cachedProductDAO.findById(savedGadget.getId());
        if (readResult.isSuccess() && readResult.getValue().isPresent()) {
            logger.info("Read by ID: {}", readResult.getValue().get());
        }
        
        // Read by SKU
        Result<Optional<Product>> skuResult = cachedProductDAO.findBySku("GDG-001");
        if (skuResult.isSuccess() && skuResult.getValue().isPresent()) {
            logger.info("Read by SKU: {}", skuResult.getValue().get());
        }
        
        // Update
        Product updatedGadget = savedGadget.withQuantity(75).withPrice(new BigDecimal("44.99"));
        Result<Product> updateResult = cachedProductDAO.save(updatedGadget);
        if (updateResult.isSuccess()) {
            logger.info("Updated: {}", updateResult.getOrThrow());
        }
        
        // List all
        Result<List<Product>> listResult = cachedProductDAO.findAll();
        if (listResult.isSuccess()) {
            logger.info("Total active products: {}", listResult.getOrThrow().size());
        }
        
        // Delete (soft delete)
        Result<Void> deleteResult = cachedProductDAO.delete(savedGadget.getId());
        if (deleteResult.isSuccess()) {
            logger.info("Deleted product id={}", savedGadget.getId());
        }
    }

    private void demonstrateCachePerformance() {
        logger.info("--- Cache Performance ---");
        
        // Create a product for testing
        Product testProduct = Product.builder()
            .sku("CACHE-001")
            .name("Cache Test Product")
            .quantity(10)
            .price(new BigDecimal("9.99"))
            .build();
        
        cachedProductDAO.save(testProduct);
        
        // First access - cache miss
        long start = System.nanoTime();
        cachedProductDAO.findBySku("CACHE-001");
        long firstAccess = System.nanoTime() - start;
        
        // Second access - cache hit
        start = System.nanoTime();
        cachedProductDAO.findBySku("CACHE-001");
        long secondAccess = System.nanoTime() - start;
        
        logger.info("First access (cache miss): {} ns", firstAccess);
        logger.info("Second access (cache hit): {} ns", secondAccess);
        logger.info("Cache stats: {}", cachedProductDAO.getCacheStats());
    }

    private void demonstrateBatchOperations() {
        logger.info("--- Batch Operations ---");
        
        List<Product> products = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            Product product = Product.builder()
                .sku(String.format("BATCH-%03d", i))
                .name(String.format("Batch Product %d", i))
                .quantity(i * 10)
                .price(new BigDecimal(String.format("%.2f", i * 1.5)))
                .build();
            products.add(product);
        }
        
        long start = System.currentTimeMillis();
        Result<Integer> result = cachedProductDAO.batchInsert(products);
        long duration = System.currentTimeMillis() - start;
        
        if (result.isSuccess()) {
            logger.info("Batch inserted {} products in {} ms", result.getOrThrow(), duration);
        } else {
            logger.error("Batch insert failed: {}", result.getError());
        }
    }

    private void demonstrateTransactions() {
        logger.info("--- Transaction Management ---");
        
        // Simulate a multi-step operation that should be atomic
        Result<String> txResult = productDAO.executeInTransaction(conn -> {
            // Step 1: Create product
            Product p1 = Product.builder()
                .sku("TX-001")
                .name("Transaction Test 1")
                .quantity(100)
                .price(new BigDecimal("10.00"))
                .build();
            
            ProductDAO txDAO = new ProductDAO(new DataSource() {
                @Override
                public Connection getConnection() { return conn; }
                @Override
                public Connection getConnection(String username, String password) { return conn; }
                @Override
                public java.io.PrintWriter getLogWriter() { return null; }
                @Override
                public void setLogWriter(java.io.PrintWriter out) {}
                @Override
                public void setLoginTimeout(int seconds) {}
                @Override
                public int getLoginTimeout() { return 0; }
                @Override
                public java.util.logging.Logger getParentLogger() { return null; }
                @Override
                public <T> T unwrap(Class<T> iface) { return null; }
                @Override
                public boolean isWrapperFor(Class<?> iface) { return false; }
            });
            Result<Product> r1 = txDAO.save(p1);
            
            if (r1.isFailure()) {
                throw new Exception("Failed to create product 1");
            }
            
            // Step 2: Create another product
            Product p2 = Product.builder()
                .sku("TX-002")
                .name("Transaction Test 2")
                .quantity(50)
                .price(new BigDecimal("15.00"))
                .build();
            
            Result<Product> r2 = txDAO.save(p2);
            
            if (r2.isFailure()) {
                throw new Exception("Failed to create product 2");
            }
            
            return "Transaction completed: 2 products created";
        });
        
        if (txResult.isSuccess()) {
            logger.info("Transaction result: {}", txResult.getOrThrow());
        } else {
            logger.error("Transaction failed: {}", txResult.getError());
        }
    }

    private void demonstrateBackup() {
        logger.info("--- Database Backup ---");
        
        boolean success = DatabaseBackup.backup("stocksync.db", "backups");
        if (success) {
            logger.info("Backup completed successfully");
            DatabaseBackup.listBackups("backups");
        } else {
            logger.error("Backup failed");
        }
    }

    public void shutdown() {
        logger.info("Shutting down StockSync application");
        if (dataSource instanceof AutoCloseable closeable) {
            try {
                closeable.close();
                logger.info("DataSource closed");
            } catch (Exception e) {
                logger.error("Error closing DataSource", e);
            }
        }
    }

    public static void main(String[] args) {
        StockSyncApp app = null;
        try {
            app = new StockSyncApp("stocksync.db");
            app.runDemo();
        } catch (Exception e) {
            logger.error("Application error", e);
            System.exit(1);
        } finally {
            if (app != null) {
                app.shutdown();
            }
        }
    }
}
