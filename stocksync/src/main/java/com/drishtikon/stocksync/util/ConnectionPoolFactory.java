package com.drishtikon.stocksync.util;

import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;

/**
 * Day 5: HikariCP connection pool configuration.
 * 
 * Connection pooling is critical for:
 * - Performance (reusing connections)
 * - Reliability (managing connection lifecycle)
 * - Resource management (limiting concurrent connections)
 */
public class ConnectionPoolFactory {
    private static final Logger logger = LoggerFactory.getLogger(ConnectionPoolFactory.class);
    
    private static final String DEFAULT_DB_URL = "jdbc:sqlite:stocksync.db";
    private static final int DEFAULT_MAX_POOL_SIZE = 10;
    private static final int DEFAULT_MIN_IDLE = 2;
    private static final long DEFAULT_CONNECTION_TIMEOUT = 30000; // 30 seconds
    private static final long DEFAULT_IDLE_TIMEOUT = 600000; // 10 minutes
    private static final long DEFAULT_MAX_LIFETIME = 1800000; // 30 minutes

    public static DataSource createDataSource() {
        return createDataSource(DEFAULT_DB_URL);
    }

    public static DataSource createDataSource(String jdbcUrl) {
        logger.info("Creating HikariCP connection pool for: {}", jdbcUrl);
        
        HikariConfig config = new HikariConfig();
        
        // JDBC URL
        config.setJdbcUrl(jdbcUrl);
        
        // Pool sizing
        config.setMaximumPoolSize(DEFAULT_MAX_POOL_SIZE);
        config.setMinimumIdle(DEFAULT_MIN_IDLE);
        
        // Timeouts
        config.setConnectionTimeout(DEFAULT_CONNECTION_TIMEOUT);
        config.setIdleTimeout(DEFAULT_IDLE_TIMEOUT);
        config.setMaxLifetime(DEFAULT_MAX_LIFETIME);
        
        // Pool name for logging
        config.setPoolName("StockSyncPool");
        
        // Connection test query
        config.setConnectionTestQuery("SELECT 1");
        
        // Auto-commit
        config.setAutoCommit(true);
        
        // Leak detection (development only)
        config.setLeakDetectionThreshold(60000); // 60 seconds
        
        HikariDataSource dataSource = new HikariDataSource(config);
        
        logger.info("HikariCP connection pool created successfully");
        return dataSource;
    }

    /**
     * Create DataSource with custom configuration
     */
    public static DataSource createDataSource(String jdbcUrl, int maxPoolSize, int minIdle) {
        logger.info("Creating HikariCP connection pool with custom config");
        
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl(jdbcUrl);
        config.setMaximumPoolSize(maxPoolSize);
        config.setMinimumIdle(minIdle);
        config.setConnectionTimeout(DEFAULT_CONNECTION_TIMEOUT);
        config.setIdleTimeout(DEFAULT_IDLE_TIMEOUT);
        config.setMaxLifetime(DEFAULT_MAX_LIFETIME);
        config.setPoolName("StockSyncPool");
        config.setConnectionTestQuery("SELECT 1");
        config.setAutoCommit(true);
        
        return new HikariDataSource(config);
    }
}
