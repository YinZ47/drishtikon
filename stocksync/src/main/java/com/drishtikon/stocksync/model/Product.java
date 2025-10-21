package com.drishtikon.stocksync.model;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Product entity with Builder pattern for clean construction.
 * Day 1-2: Demonstrates Builder pattern, immutability, and encapsulation.
 * 
 * Immutable product class - all modifications return new instances.
 */
public final class Product {
    private final Long id;
    private final String sku;
    private final String name;
    private final int quantity;
    private final BigDecimal price;
    private final LocalDateTime createdAt;
    private final LocalDateTime updatedAt;
    private final boolean deleted;

    private Product(Builder builder) {
        this.id = builder.id;
        this.sku = Objects.requireNonNull(builder.sku, "SKU cannot be null");
        this.name = Objects.requireNonNull(builder.name, "Name cannot be null");
        this.quantity = builder.quantity;
        this.price = Objects.requireNonNull(builder.price, "Price cannot be null");
        this.createdAt = builder.createdAt;
        this.updatedAt = builder.updatedAt;
        this.deleted = builder.deleted;
        
        validateProduct();
    }

    private void validateProduct() {
        if (sku.isBlank()) {
            throw new IllegalArgumentException("SKU cannot be blank");
        }
        if (name.isBlank()) {
            throw new IllegalArgumentException("Name cannot be blank");
        }
        if (quantity < 0) {
            throw new IllegalArgumentException("Quantity cannot be negative");
        }
        if (price.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Price cannot be negative");
        }
    }

    // Getters
    public Long getId() {
        return id;
    }

    public String getSku() {
        return sku;
    }

    public String getName() {
        return name;
    }

    public int getQuantity() {
        return quantity;
    }

    public BigDecimal getPrice() {
        return price;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public boolean isDeleted() {
        return deleted;
    }

    // Copy methods for immutability
    public Product withQuantity(int newQuantity) {
        return new Builder(this).quantity(newQuantity).build();
    }

    public Product withPrice(BigDecimal newPrice) {
        return new Builder(this).price(newPrice).build();
    }

    public Product markAsDeleted() {
        return new Builder(this).deleted(true).build();
    }

    public Product withUpdatedAt(LocalDateTime updatedAt) {
        return new Builder(this).updatedAt(updatedAt).build();
    }

    // Builder Pattern
    public static class Builder {
        private Long id;
        private String sku;
        private String name;
        private int quantity;
        private BigDecimal price;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;
        private boolean deleted;

        public Builder() {
            this.createdAt = LocalDateTime.now();
            this.updatedAt = LocalDateTime.now();
            this.deleted = false;
        }

        // Copy constructor for creating modified versions
        public Builder(Product product) {
            this.id = product.id;
            this.sku = product.sku;
            this.name = product.name;
            this.quantity = product.quantity;
            this.price = product.price;
            this.createdAt = product.createdAt;
            this.updatedAt = product.updatedAt;
            this.deleted = product.deleted;
        }

        public Builder id(Long id) {
            this.id = id;
            return this;
        }

        public Builder sku(String sku) {
            this.sku = sku;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder quantity(int quantity) {
            this.quantity = quantity;
            return this;
        }

        public Builder price(BigDecimal price) {
            this.price = price;
            return this;
        }

        public Builder createdAt(LocalDateTime createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder updatedAt(LocalDateTime updatedAt) {
            this.updatedAt = updatedAt;
            return this;
        }

        public Builder deleted(boolean deleted) {
            this.deleted = deleted;
            return this;
        }

        public Product build() {
            return new Product(this);
        }
    }

    // Static factory method
    public static Builder builder() {
        return new Builder();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Product product = (Product) o;
        return Objects.equals(id, product.id) &&
               Objects.equals(sku, product.sku);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, sku);
    }

    @Override
    public String toString() {
        return "Product{" +
                "id=" + id +
                ", sku='" + sku + '\'' +
                ", name='" + name + '\'' +
                ", quantity=" + quantity +
                ", price=" + price +
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                ", deleted=" + deleted +
                '}';
    }
}
