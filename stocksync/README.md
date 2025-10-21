# StockSync Database Sprint - 10-Day Java/JDBC/SQLite Learning Project

A comprehensive hands-on project implementing a complete database-backed inventory management system using Java, JDBC, SQLite, and modern best practices.

## 🎯 Project Overview

This project demonstrates the complete 10-day StockSync Database Sprint roadmap, implementing:

- **Day 1-2**: Java essentials (Builder pattern, generics, immutability)
- **Day 3**: SQLite schema design with triggers and indexes
- **Day 4**: JDBC fundamentals with full CRUD operations
- **Day 5**: HikariCP connection pooling and transaction management
- **Day 6**: In-memory write-through cache layer
- **Day 7**: Performance optimization and benchmarking
- **Day 8**: Batch operations for bulk data handling
- **Day 9**: Error handling, logging with SLF4J/Logback
- **Day 10**: Health checks, backup/restore utilities

## 📋 Prerequisites

- Java 17 or higher
- Maven 3.6 or higher
- Basic understanding of SQL

## 🚀 Quick Start

### 1. Build the Project

```bash
cd stocksync
mvn clean compile
```

### 2. Run Tests

```bash
mvn test
```

### 3. Run the Application

```bash
mvn exec:java -Dexec.mainClass="com.drishtikon.stocksync.StockSyncApp"
```

Or compile and run:

```bash
mvn clean package
java -jar target/stocksync-1.0.0.jar
```

## 📁 Project Structure

```
stocksync/
├── src/
│   ├── main/
│   │   ├── java/com/drishtikon/stocksync/
│   │   │   ├── model/
│   │   │   │   └── Product.java              # Domain model with Builder pattern
│   │   │   ├── dao/
│   │   │   │   ├── SchemaInitializer.java    # Database schema setup
│   │   │   │   └── ProductDAO.java            # JDBC CRUD operations
│   │   │   ├── cache/
│   │   │   │   └── CachedProductDAO.java      # Write-through cache layer
│   │   │   ├── util/
│   │   │   │   ├── Result.java                # Generic result wrapper
│   │   │   │   ├── ConnectionPoolFactory.java # HikariCP setup
│   │   │   │   └── DatabaseBackup.java        # Backup/restore utilities
│   │   │   ├── health/
│   │   │   │   └── DatabaseHealthCheck.java   # Health monitoring
│   │   │   └── StockSyncApp.java             # Main application
│   │   └── resources/
│   │       └── logback.xml                    # Logging configuration
│   └── test/
│       └── java/com/drishtikon/stocksync/
│           ├── model/
│           │   └── ProductTest.java           # Product tests
│           └── util/
│               └── ResultTest.java            # Result wrapper tests
├── pom.xml                                     # Maven configuration
└── README.md                                   # This file
```

## 🗄️ Database Schema

### Products Table

```sql
CREATE TABLE products (
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
);

-- Indexes for performance
CREATE INDEX idx_products_sku ON products(sku) WHERE deleted = 0;
CREATE INDEX idx_products_active ON products(deleted) WHERE deleted = 0;

-- Trigger for auto-updating timestamps
CREATE TRIGGER update_products_timestamp 
AFTER UPDATE ON products
FOR EACH ROW
BEGIN
    UPDATE products SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
```

## 🔑 Key Features & Learning Points

### 1. Builder Pattern & Immutability (Day 1-2)

```java
Product product = Product.builder()
    .sku("WDG-001")
    .name("Super Widget")
    .quantity(100)
    .price(new BigDecimal("29.99"))
    .build();

// Immutable modifications return new instances
Product updated = product.withQuantity(150);
```

### 2. Generic Result Wrapper (Day 1-2)

```java
Result<Product> result = productDAO.findById(1L);
if (result.isSuccess()) {
    Product product = result.getOrThrow();
    // Use product
} else {
    logger.error("Error: {}", result.getError());
}
```

### 3. JDBC with PreparedStatements (Day 4)

```java
String sql = "SELECT * FROM products WHERE sku = ?";
try (Connection conn = dataSource.getConnection();
     PreparedStatement pstmt = conn.prepareStatement(sql)) {
    pstmt.setString(1, sku);
    try (ResultSet rs = pstmt.executeQuery()) {
        // Process results
    }
}
```

**Important**: Always use `PreparedStatement` for user input to prevent SQL injection!

### 4. Connection Pooling with HikariCP (Day 5)

```java
DataSource dataSource = ConnectionPoolFactory.createDataSource("jdbc:sqlite:stocksync.db");
// Connection pooling handles resource management automatically
```

**Benefits**:
- Reuses connections instead of creating new ones
- Configurable pool size limits concurrent connections
- Automatic connection validation and health checks

### 5. Transaction Management (Day 5)

```java
Result<String> txResult = productDAO.executeInTransaction(conn -> {
    // Multiple DB operations
    // If any fails, entire transaction rolls back
    return "Success";
});
```

### 6. Write-Through Cache (Day 6)

```java
// Cache lookup: DB first, then cache
Result<Optional<Product>> result = cachedProductDAO.findBySku("WDG-001");

// Cache statistics
CacheStats stats = cachedProductDAO.getCacheStats();
logger.info("Cache hit rate: {}%", stats.getHitRate());
```

**Strategy**: Write to database first, then update cache. Database is always the source of truth.

### 7. Batch Operations (Day 8)

```java
List<Product> products = createManyProducts();
Result<Integer> result = cachedProductDAO.batchInsert(products);
// 10-50x faster than individual inserts
```

### 8. Structured Logging (Day 9)

```java
private static final Logger logger = LoggerFactory.getLogger(ProductDAO.class);

logger.info("Inserted product with id={}, sku={}", id, sku);
logger.error("Database error", exception);
```

### 9. Health Checks (Day 10)

```java
DatabaseHealthCheck healthCheck = new DatabaseHealthCheck(dataSource);
HealthStatus status = healthCheck.getHealthStatus();
logger.info("Health: {}", status);
```

### 10. Backup & Restore (Day 10)

```java
// Create backup
DatabaseBackup.backup("stocksync.db", "backups");

// Restore from backup
DatabaseBackup.restore("backups/backup.db", "stocksync.db");
```

## 🧪 Running Tests

```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=ProductTest

# Run with verbose output
mvn test -X
```

## 📊 Performance Benchmarking

The application includes built-in performance demonstrations:

1. **Cache Performance**: Compares cache hit vs miss times
2. **Batch Operations**: Shows speed difference between batch and single inserts
3. **Connection Pooling**: Benchmarks connection acquisition times

## 🔍 Troubleshooting

### Common Issues

1. **"Database is locked" error**
   - SQLite doesn't handle concurrent writes well
   - Use transactions for multi-step operations
   - Consider connection pool size

2. **"No suitable driver found"**
   - Ensure sqlite-jdbc dependency is in pom.xml
   - Run `mvn clean install` to download dependencies

3. **OutOfMemoryError**
   - Check connection pool configuration
   - Ensure connections are closed (use try-with-resources)
   - Review cache size for large datasets

4. **Tests failing**
   - Delete `stocksync.db` file if it exists from previous runs
   - Check logs in `logs/stocksync.log`

### Database File Location

The SQLite database file (`stocksync.db`) is created in the current working directory. To specify a different location:

```java
StockSyncApp app = new StockSyncApp("/path/to/database.db");
```

## 📝 Best Practices Demonstrated

1. ✅ **Always use try-with-resources** for JDBC connections
2. ✅ **Always use PreparedStatement** for user input (SQL injection prevention)
3. ✅ **Never swallow exceptions** - always log them
4. ✅ **Validate input** before database operations
5. ✅ **Use connection pooling** in production
6. ✅ **Database is source of truth** - cache is for speed only
7. ✅ **Use transactions** for multi-step operations
8. ✅ **Benchmark before optimizing** - measure actual bottlenecks
9. ✅ **Structured logging** with SLF4J
10. ✅ **Immutable domain objects** with Builder pattern

## 🚨 Security Notes

- **SQL Injection Prevention**: All queries use parameterized PreparedStatements
- **No hardcoded credentials**: SQLite is file-based (no authentication)
- **Input Validation**: All user inputs validated before DB operations
- **Error Messages**: Don't expose sensitive DB details in logs

## 📚 Learning Resources

- [Java OOP Basics](https://docs.oracle.com/javase/tutorial/java/concepts)
- [JDBC Tutorial](https://docs.oracle.com/javase/tutorial/jdbc/basics/index.html)
- [PreparedStatement API](https://docs.oracle.com/javase/8/docs/api/java/sql/PreparedStatement.html)
- [HikariCP Wiki](https://github.com/brettwooldridge/HikariCP/wiki/Configuration)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [SLF4J Manual](http://www.slf4j.org/manual.html)

## 🤝 Contributing

This is a learning project. Feel free to:
- Add more features (e.g., queries, aggregations)
- Improve error handling
- Add more comprehensive tests
- Optimize performance further

## 📄 License

This project is part of the Drishtikon repository and follows its licensing terms.

## 🎓 Daily Sprint Checklist

Track your progress through the 10-day sprint:

- [ ] Day 1-2: Completed Product class with Builder, Result wrapper, tests
- [ ] Day 3: Created SQLite schema with indexes and triggers
- [ ] Day 4: Implemented full CRUD with ProductDAO
- [ ] Day 5: Added HikariCP pooling and transaction support
- [ ] Day 6: Built write-through cache layer
- [ ] Day 7: Benchmarked and optimized queries
- [ ] Day 8: Implemented batch operations
- [ ] Day 9: Added logging and validation
- [ ] Day 10: Created health checks, backup utilities, and documentation

## 💡 Next Steps

After completing this sprint, consider:

1. **Add REST API layer** with Spring Boot
2. **Implement search functionality** with full-text search
3. **Add pagination** for large result sets
4. **Create audit logging** for compliance
5. **Add reporting features** with aggregations
6. **Implement user authentication** and authorization
7. **Create a web UI** for inventory management
8. **Add metrics and monitoring** with Micrometer

---

**Remember**: The best way to learn is by doing. Run the code, break it, fix it, and make it your own!
