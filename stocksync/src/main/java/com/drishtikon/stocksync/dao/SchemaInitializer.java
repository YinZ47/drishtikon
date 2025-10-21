package com.drishtikon.stocksync.dao;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * Day 3: SQLite schema initialization with triggers and indexes.
 * 
 * Creates the products table with:
 * - Proper indexes on SKU and active products
 * - Triggers for auto-updating timestamps
 * - Soft delete support
 */
public class SchemaInitializer {
    private static final Logger logger = LoggerFactory.getLogger(SchemaInitializer.class);

    private static final String CREATE_PRODUCTS_TABLE = """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 0,
            price REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            deleted INTEGER NOT NULL DEFAULT 0,
            CHECK (quantity >= 0),
            CHECK (price >= 0),
            CHECK (deleted IN (0, 1))
        )
        """;

    private static final String CREATE_SKU_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_products_sku 
        ON products(sku) WHERE deleted = 0
        """;

    private static final String CREATE_ACTIVE_PRODUCTS_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_products_active 
        ON products(deleted) WHERE deleted = 0
        """;

    private static final String CREATE_UPDATE_TRIGGER = """
        CREATE TRIGGER IF NOT EXISTS update_products_timestamp 
        AFTER UPDATE ON products
        FOR EACH ROW
        BEGIN
            UPDATE products SET updated_at = CURRENT_TIMESTAMP 
            WHERE id = NEW.id;
        END
        """;

    public static void initializeSchema(Connection connection) throws SQLException {
        logger.info("Initializing database schema...");
        
        try (Statement stmt = connection.createStatement()) {
            // Create table
            stmt.execute(CREATE_PRODUCTS_TABLE);
            logger.info("Products table created or already exists");
            
            // Create indexes
            stmt.execute(CREATE_SKU_INDEX);
            logger.info("SKU index created");
            
            stmt.execute(CREATE_ACTIVE_PRODUCTS_INDEX);
            logger.info("Active products index created");
            
            // Create trigger
            stmt.execute(CREATE_UPDATE_TRIGGER);
            logger.info("Update timestamp trigger created");
            
            logger.info("Schema initialization completed successfully");
        } catch (SQLException e) {
            logger.error("Failed to initialize schema", e);
            throw e;
        }
    }
}
