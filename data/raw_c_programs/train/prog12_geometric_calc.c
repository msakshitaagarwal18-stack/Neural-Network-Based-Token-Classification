/* Program 12: Geometric Calculations and Coordinates */
struct Point {
    double x;
    double y;
};

struct Rectangle {
    struct Point top_left;
    struct Point bottom_right;
};

double distance_squared(struct Point p1, struct Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

double rectangle_area(struct Rectangle rect) {
    double width = rect.bottom_right.x - rect.top_left.x;
    double height = rect.top_left.y - rect.bottom_right.y;
    if (width < 0.0) width = -width;
    if (height < 0.0) height = -height;
    return width * height;
}

int point_in_rect(struct Point p, struct Rectangle r) {
    if (p.x >= r.top_left.x && p.x <= r.bottom_right.x &&
        p.y <= r.top_left.y && p.y >= r.bottom_right.y) {
        return 1;
    }
    return 0;
}
