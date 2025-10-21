package com.drishtikon.stocksync.model;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Product class - Day 1-2
 */
class ProductTest {

    @Test
    void testBuilderCreatesValidProduct() {
        Product product = Product.builder()
            .sku("TEST-001")
            .name("Test Product")
            .quantity(10)
            .price(new BigDecimal("19.99"))
            .build();

        assertNotNull(product);
        assertEquals("TEST-001", product.getSku());
        assertEquals("Test Product", product.getName());
        assertEquals(10, product.getQuantity());
        assertEquals(new BigDecimal("19.99"), product.getPrice());
        assertFalse(product.isDeleted());
    }

    @Test
    void testBuilderFailsWithNullSku() {
        assertThrows(NullPointerException.class, () -> {
            Product.builder()
                .name("Test Product")
                .quantity(10)
                .price(new BigDecimal("19.99"))
                .build();
        });
    }

    @Test
    void testBuilderFailsWithBlankSku() {
        assertThrows(IllegalArgumentException.class, () -> {
            Product.builder()
                .sku("")
                .name("Test Product")
                .quantity(10)
                .price(new BigDecimal("19.99"))
                .build();
        });
    }

    @Test
    void testBuilderFailsWithNegativeQuantity() {
        assertThrows(IllegalArgumentException.class, () -> {
            Product.builder()
                .sku("TEST-001")
                .name("Test Product")
                .quantity(-1)
                .price(new BigDecimal("19.99"))
                .build();
        });
    }

    @Test
    void testBuilderFailsWithNegativePrice() {
        assertThrows(IllegalArgumentException.class, () -> {
            Product.builder()
                .sku("TEST-001")
                .name("Test Product")
                .quantity(10)
                .price(new BigDecimal("-1.00"))
                .build();
        });
    }

    @Test
    void testImmutabilityWithQuantity() {
        Product original = Product.builder()
            .sku("TEST-001")
            .name("Test Product")
            .quantity(10)
            .price(new BigDecimal("19.99"))
            .build();

        Product modified = original.withQuantity(20);

        assertNotSame(original, modified);
        assertEquals(10, original.getQuantity());
        assertEquals(20, modified.getQuantity());
    }

    @Test
    void testImmutabilityWithPrice() {
        Product original = Product.builder()
            .sku("TEST-001")
            .name("Test Product")
            .quantity(10)
            .price(new BigDecimal("19.99"))
            .build();

        Product modified = original.withPrice(new BigDecimal("29.99"));

        assertNotSame(original, modified);
        assertEquals(new BigDecimal("19.99"), original.getPrice());
        assertEquals(new BigDecimal("29.99"), modified.getPrice());
    }

    @Test
    void testMarkAsDeleted() {
        Product product = Product.builder()
            .sku("TEST-001")
            .name("Test Product")
            .quantity(10)
            .price(new BigDecimal("19.99"))
            .build();

        Product deleted = product.markAsDeleted();

        assertFalse(product.isDeleted());
        assertTrue(deleted.isDeleted());
    }

    @Test
    void testEquality() {
        Product p1 = Product.builder()
            .id(1L)
            .sku("TEST-001")
            .name("Test Product")
            .quantity(10)
            .price(new BigDecimal("19.99"))
            .build();

        Product p2 = Product.builder()
            .id(1L)
            .sku("TEST-001")
            .name("Different Name")
            .quantity(20)
            .price(new BigDecimal("29.99"))
            .build();

        assertEquals(p1, p2); // Same ID and SKU
        assertEquals(p1.hashCode(), p2.hashCode());
    }
}
