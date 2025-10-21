package com.drishtikon.stocksync.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Day 10: SQLite database backup and restore utility.
 * 
 * Simple file-based backup for SQLite databases.
 */
public class DatabaseBackup {
    private static final Logger logger = LoggerFactory.getLogger(DatabaseBackup.class);
    private static final DateTimeFormatter TIMESTAMP_FORMAT = 
        DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    /**
     * Create a backup of the database file
     */
    public static boolean backup(String dbPath, String backupDir) {
        try {
            Path sourcePath = Paths.get(dbPath);
            
            if (!Files.exists(sourcePath)) {
                logger.error("Database file does not exist: {}", dbPath);
                return false;
            }

            // Create backup directory if it doesn't exist
            Path backupDirPath = Paths.get(backupDir);
            Files.createDirectories(backupDirPath);

            // Generate backup filename with timestamp
            String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
            String backupFileName = String.format("stocksync_backup_%s.db", timestamp);
            Path backupPath = backupDirPath.resolve(backupFileName);

            // Copy database file
            Files.copy(sourcePath, backupPath, StandardCopyOption.REPLACE_EXISTING);

            logger.info("Database backed up to: {}", backupPath);
            return true;

        } catch (IOException e) {
            logger.error("Failed to backup database", e);
            return false;
        }
    }

    /**
     * Restore database from a backup file
     */
    public static boolean restore(String backupPath, String dbPath) {
        try {
            Path sourcePath = Paths.get(backupPath);
            Path targetPath = Paths.get(dbPath);

            if (!Files.exists(sourcePath)) {
                logger.error("Backup file does not exist: {}", backupPath);
                return false;
            }

            // Create backup of current DB before restore
            if (Files.exists(targetPath)) {
                String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
                Path preRestoreBackup = Paths.get(dbPath + ".pre_restore_" + timestamp);
                Files.copy(targetPath, preRestoreBackup, StandardCopyOption.REPLACE_EXISTING);
                logger.info("Created pre-restore backup: {}", preRestoreBackup);
            }

            // Restore from backup
            Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);

            logger.info("Database restored from: {}", backupPath);
            return true;

        } catch (IOException e) {
            logger.error("Failed to restore database", e);
            return false;
        }
    }

    /**
     * List all backup files in the backup directory
     */
    public static void listBackups(String backupDir) {
        try {
            Path backupDirPath = Paths.get(backupDir);
            
            if (!Files.exists(backupDirPath)) {
                logger.info("No backup directory found: {}", backupDir);
                return;
            }

            logger.info("Available backups in {}:", backupDir);
            Files.list(backupDirPath)
                .filter(path -> path.toString().endsWith(".db"))
                .forEach(path -> logger.info("  - {}", path.getFileName()));

        } catch (IOException e) {
            logger.error("Failed to list backups", e);
        }
    }
}
