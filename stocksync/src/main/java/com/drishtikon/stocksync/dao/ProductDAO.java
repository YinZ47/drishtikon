package com.drishtikon.stocksync.dao;

import com.drishtikon.stocksync.model.Product;
import com.drishtikon.stocksync.util.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.math.BigDecimal;
import java.sql.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Day 4: JDBC-based ProductDAO with full CRUD operations.
 * Day 5: Refactored to use DataSource for connection pooling.
 * Day 8: Added batch operations support.
 * Day 9: Enhanced with validation and structured logging.
 * 
 * Always uses PreparedStatement to prevent SQL injection.
 * Uses try-with-resources for automatic resource management.
 */
public class ProductDAO {
    private static final Logger logger = LoggerFactory.getLogger(ProductDAO.class);
    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
    
    private final DataSource dataSource;

    public ProductDAO(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    /**
     * Find product by ID
     */
    public Result<Optional<Product>> findById(Long id) {
        if (id == null || id <= 0) {
            return Result.failure("Invalid product ID");
        }

        String sql = "SELECT * FROM products WHERE id = ? AND deleted = 0";
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setLong(1, id);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    Product product = mapResultSetToProduct(rs);
                    logger.debug("Found product with id={}", id);
                    return Result.success(Optional.of(product));
                }
                logger.debug("No product found with id={}", id);
                return Result.success(Optional.empty());
            }
            
        } catch (SQLException e) {
            logger.error("Error finding product by id={}", id, e);
            return Result.failure("Database error: " + e.getMessage());
        }
    }

    /**
     * Find product by SKU
     */
    public Result<Optional<Product>> findBySku(String sku) {
        if (sku == null || sku.isBlank()) {
            return Result.failure("Invalid SKU");
        }

        String sql = "SELECT * FROM products WHERE sku = ? AND deleted = 0";
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setString(1, sku);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    Product product = mapResultSetToProduct(rs);
                    logger.debug("Found product with sku={}", sku);
                    return Result.success(Optional.of(product));
                }
                logger.debug("No product found with sku={}", sku);
                return Result.success(Optional.empty());
            }
            
        } catch (SQLException e) {
            logger.error("Error finding product by sku={}", sku, e);
            return Result.failure("Database error: " + e.getMessage());
        }
    }

    /**
     * Find all active products
     */
    public Result<List<Product>> findAll() {
        String sql = "SELECT * FROM products WHERE deleted = 0 ORDER BY sku";
        List<Product> products = new ArrayList<>();
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql);
             ResultSet rs = pstmt.executeQuery()) {
            
            while (rs.next()) {
                products.add(mapResultSetToProduct(rs));
            }
            
            logger.info("Found {} active products", products.size());
            return Result.success(products);
            
        } catch (SQLException e) {
            logger.error("Error finding all products", e);
            return Result.failure("Database error: " + e.getMessage());
        }
    }

    /**
     * Save (insert or update) a product
     */
    public Result<Product> save(Product product) {
        if (product == null) {
            return Result.failure("Product cannot be null");
        }

        if (product.getId() == null) {
            return insert(product);
        } else {
            return update(product);
        }
    }

    /**
     * Insert a new product
     */
    private Result<Product> insert(Product product) {
        String sql = """
            INSERT INTO products (sku, name, quantity, price, created_at, updated_at, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """;
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS)) {
            
            pstmt.setString(1, product.getSku());
            pstmt.setString(2, product.getName());
            pstmt.setInt(3, product.getQuantity());
            pstmt.setBigDecimal(4, product.getPrice());
            pstmt.setString(5, formatDateTime(product.getCreatedAt()));
            pstmt.setString(6, formatDateTime(product.getUpdatedAt()));
            pstmt.setInt(7, product.isDeleted() ? 1 : 0);
            
            int affectedRows = pstmt.executeUpdate();
            
            if (affectedRows == 0) {
                return Result.failure("Insert failed, no rows affected");
            }
            
            try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                if (generatedKeys.next()) {
                    long id = generatedKeys.getLong(1);
                    Product savedProduct = Product.builder()
                        .id(id)
                        .sku(product.getSku())
                        .name(product.getName())
                        .quantity(product.getQuantity())
                        .price(product.getPrice())
                        .createdAt(product.getCreatedAt())
                        .updatedAt(product.getUpdatedAt())
                        .deleted(product.isDeleted())
                        .build();
                    
                    logger.info("Inserted product with id={}, sku={}", id, product.getSku());
                    return Result.success(savedProduct);
                } else {
                    return Result.failure("Insert failed, no ID obtained");
                }
            }
            
        } catch (SQLException e) {
            logger.error("Error inserting product sku={}", product.getSku(), e);
            return Result.failure("Database error: " + e.getMessage());
        }
    }

    /**
     * Update an existing product
     */
    private Result<Product> update(Product product) {
        String sql = """
            UPDATE products 
            SET sku = ?, name = ?, quantity = ?, price = ?, deleted = ?
            WHERE id = ?
            """;
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setString(1, product.getSku());
            pstmt.setString(2, product.getName());
            pstmt.setInt(3, product.getQuantity());
            pstmt.setBigDecimal(4, product.getPrice());
            pstmt.setInt(5, product.isDeleted() ? 1 : 0);
            pstmt.setLong(6, product.getId());
            
            int affectedRows = pstmt.executeUpdate();
            
            if (affectedRows == 0) {
                return Result.failure("Update failed, product not found");
            }
            
            logger.info("Updated product with id={}, sku={}", product.getId(), product.getSku());
            return Result.success(product);
            
        } catch (SQLException e) {
            logger.error("Error updating product id={}", product.getId(), e);
            return Result.failure("Database error: " + e.getMessage());
        }
    }

    /**
     * Soft delete a product by ID
     */
    public Result<Void> delete(Long id) {
        if (id == null || id <= 0) {
            return Result.failure("Invalid product ID");
        }

        String sql = "UPDATE products SET deleted = 1 WHERE id = ?";
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setLong(1, id);
            int affectedRows = pstmt.executeUpdate();
            
            if (affectedRows == 0) {
                return Result.failure("Delete failed, product not found");
            }
            
            logger.info("Soft deleted product with id={}", id);
            return Result.success(null);
            
        } catch (SQLException e) {
            logger.error("Error deleting product id={}", id, e);
            return Result.failure("Database error: " + e.getMessage());
        }
    }

    /**
     * Day 8: Batch insert products
     */
    public Result<Integer> batchInsert(List<Product> products) {
        if (products == null || products.isEmpty()) {
            return Result.failure("Product list cannot be empty");
        }

        String sql = """
            INSERT INTO products (sku, name, quantity, price, created_at, updated_at, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """;
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            conn.setAutoCommit(false);
            
            for (Product product : products) {
                pstmt.setString(1, product.getSku());
                pstmt.setString(2, product.getName());
                pstmt.setInt(3, product.getQuantity());
                pstmt.setBigDecimal(4, product.getPrice());
                pstmt.setString(5, formatDateTime(product.getCreatedAt()));
                pstmt.setString(6, formatDateTime(product.getUpdatedAt()));
                pstmt.setInt(7, product.isDeleted() ? 1 : 0);
                pstmt.addBatch();
            }
            
            int[] results = pstmt.executeBatch();
            conn.commit();
            
            int inserted = results.length;
            logger.info("Batch inserted {} products", inserted);
            return Result.success(inserted);
            
        } catch (SQLException e) {
            logger.error("Error in batch insert", e);
            return Result.failure("Database error: " + e.getMessage());
        }
    }

    /**
     * Day 5: Execute operation within a transaction
     */
    public <T> Result<T> executeInTransaction(TransactionCallback<T> callback) {
        try (Connection conn = dataSource.getConnection()) {
            conn.setAutoCommit(false);
            
            try {
                T result = callback.execute(conn);
                conn.commit();
                logger.debug("Transaction committed successfully");
                return Result.success(result);
            } catch (Exception e) {
                conn.rollback();
                logger.error("Transaction rolled back due to error", e);
                return Result.failure("Transaction failed: " + e.getMessage());
            }
        } catch (SQLException e) {
            logger.error("Error managing transaction", e);
            return Result.failure("Transaction error: " + e.getMessage());
        }
    }

    @FunctionalInterface
    public interface TransactionCallback<T> {
        T execute(Connection conn) throws Exception;
    }

    /**
     * Map ResultSet to Product
     */
    private Product mapResultSetToProduct(ResultSet rs) throws SQLException {
        return Product.builder()
            .id(rs.getLong("id"))
            .sku(rs.getString("sku"))
            .name(rs.getString("name"))
            .quantity(rs.getInt("quantity"))
            .price(rs.getBigDecimal("price"))
            .createdAt(parseDateTime(rs.getString("created_at")))
            .updatedAt(parseDateTime(rs.getString("updated_at")))
            .deleted(rs.getInt("deleted") == 1)
            .build();
    }

    private String formatDateTime(LocalDateTime dateTime) {
        return dateTime != null ? dateTime.format(FORMATTER) : LocalDateTime.now().format(FORMATTER);
    }

    private LocalDateTime parseDateTime(String dateTime) {
        try {
            return LocalDateTime.parse(dateTime, FORMATTER);
        } catch (Exception e) {
            logger.warn("Failed to parse datetime: {}, using current time", dateTime);
            return LocalDateTime.now();
        }
    }
}
