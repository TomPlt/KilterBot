SELECT 
    climbs.uuid, climbs.layout_id, climbs.setter_id, climbs.setter_username, climbs.name, 
    climbs.frames, climbs.edge_bottom, climbs.edge_top, climbs.edge_left, climbs.edge_right,
    climb_stats.angle, climb_stats.display_difficulty, climb_stats.benchmark_difficulty,
    beta_links.link, beta_links.foreign_username, beta_links.angle as beta_angle, 
    climb_cache_fields.ascensionist_count, climb_cache_fields.display_difficulty, climb_cache_fields.quality_average climb_stats
FROM 
    climbs
JOIN 
    climb_stats ON climbs.uuid = climb_stats.climb_uuid
JOIN 
    beta_links ON climbs.uuid = beta_links.climb_uuid
JOIN 
    climb_cache_fields ON climbs.uuid = climb_cache_fields.climb_uuid
GROUP BY 
    climbs.uuid
ORDER BY
    climb_cache_fields.ascensionist_count
DESC

