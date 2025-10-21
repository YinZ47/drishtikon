package com.drishtikon.stocksync.health;

import com.zaxxer.hikari.HikariDataSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.SQLException;

/**
 * Day 10: Health check for database and connection pool.
 * 
 * Monitors:
 * - Database connectivity
 * - Connection pool metrics
 * - Connection validity
 */
public class DatabaseHealthCheck {
    private static final Logger logger = LoggerFactory.getLogger(DatabaseHealthCheck.class);
    
    private final DataSource dataSource;

    public DatabaseHealthCheck(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    /**
     * Check if database is healthy
     */
    public boolean isHealthy() {
        try (Connection conn = dataSource.getConnection()) {
            boolean valid = conn.isValid(5); // 5 second timeout
            
            if (valid) {
                logger.debug("Database health check passed");
            } else {
                logger.warn("Database health check failed - connection not valid");
            }
            
            return valid;
        } catch (SQLException e) {
            logger.error("Database health check failed", e);
            return false;
        }
    }

    /**
     * Get detailed health status
     */
    public HealthStatus getHealthStatus() {
        long startTime = System.currentTimeMillis();
        boolean healthy = isHealthy();
        long responseTime = System.currentTimeMillis() - startTime;
        
        PoolMetrics poolMetrics = getPoolMetrics();
        
        return new HealthStatus(healthy, responseTime, poolMetrics);
    }

    /**
     * Get connection pool metrics (HikariCP specific)
     */
    private PoolMetrics getPoolMetrics() {
        if (dataSource instanceof HikariDataSource hikariDS) {
            return new PoolMetrics(
                hikariDS.getHikariPoolMXBean().getActiveConnections(),
                hikariDS.getHikariPoolMXBean().getIdleConnections(),
                hikariDS.getHikariPoolMXBean().getTotalConnections(),
                hikariDS.getHikariPoolMXBean().getThreadsAwaitingConnection()
            );
        }
        return new PoolMetrics(0, 0, 0, 0);
    }

    public static class HealthStatus {
        private final boolean healthy;
        private final long responseTimeMs;
        private final PoolMetrics poolMetrics;

        public HealthStatus(boolean healthy, long responseTimeMs, PoolMetrics poolMetrics) {
            this.healthy = healthy;
            this.responseTimeMs = responseTimeMs;
            this.poolMetrics = poolMetrics;
        }

        public boolean isHealthy() {
            return healthy;
        }

        public long getResponseTimeMs() {
            return responseTimeMs;
        }

        public PoolMetrics getPoolMetrics() {
            return poolMetrics;
        }

        @Override
        public String toString() {
            return String.format("HealthStatus{healthy=%s, responseTime=%dms, %s}", 
                healthy, responseTimeMs, poolMetrics);
        }
    }

    public static class PoolMetrics {
        private final int activeConnections;
        private final int idleConnections;
        private final int totalConnections;
        private final int threadsAwaiting;

        public PoolMetrics(int activeConnections, int idleConnections, 
                          int totalConnections, int threadsAwaiting) {
            this.activeConnections = activeConnections;
            this.idleConnections = idleConnections;
            this.totalConnections = totalConnections;
            this.threadsAwaiting = threadsAwaiting;
        }

        public int getActiveConnections() {
            return activeConnections;
        }

        public int getIdleConnections() {
            return idleConnections;
        }

        public int getTotalConnections() {
            return totalConnections;
        }

        public int getThreadsAwaiting() {
            return threadsAwaiting;
        }

        @Override
        public String toString() {
            return String.format("PoolMetrics{active=%d, idle=%d, total=%d, awaiting=%d}", 
                activeConnections, idleConnections, totalConnections, threadsAwaiting);
        }
    }
}
