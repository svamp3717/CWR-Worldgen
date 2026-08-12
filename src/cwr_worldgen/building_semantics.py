# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


_CHRISTIAN_BUILDING_VALUES = frozenset({"church", "chapel", "cathedral"})
_NON_CHURCH_WORSHIP_BUILDINGS = frozenset({"mosque", "synagogue", "temple", "shrine"})


def is_actual_church(tags: Mapping[str, str]) -> bool:
    """Return true only for explicitly Christian church buildings.

    A generic ``amenity=place_of_worship`` is not enough. OSM uses that tag for
    mosques, synagogues, temples, meeting halls, and many other buildings that
    should not receive a Christian tower-and-spire model.
    """

    building = str(tags.get("building", "")).casefold()
    if building in _CHRISTIAN_BUILDING_VALUES:
        return True
    if building in _NON_CHURCH_WORSHIP_BUILDINGS:
        return False
    amenity = str(tags.get("amenity", "")).casefold()
    religion = str(tags.get("religion", "")).casefold()
    return amenity == "place_of_worship" and religion == "christian"


@dataclass(frozen=True, slots=True)
class RegionProfile:
    identifier: str
    display_name: str
    polygon_lon_lat: tuple[tuple[float, float], ...] = ()
    envelopes_lon_lat: tuple[tuple[float, float, float, float], ...] = ()
    country_aliases: frozenset[str] = frozenset()


# A deliberately lightweight, dependency-free outline. It is only used to pick
# an architectural palette, never for legal or cartographic boundary claims.
_SWEDEN = RegionProfile(
    identifier="sweden",
    display_name="Sweden",
    country_aliases=frozenset({"se", "swe", "sweden", "sverige"}),
    polygon_lon_lat=(
        (12.45, 55.20), (14.70, 55.25), (16.30, 56.20), (16.85, 57.70),
        (18.90, 59.20), (18.75, 60.60), (17.20, 61.80), (18.60, 63.10),
        (21.60, 64.60), (23.80, 65.80), (23.60, 67.20), (21.00, 69.15),
        (17.20, 69.10), (15.30, 67.30), (14.00, 65.30), (12.60, 62.80),
        (11.00, 59.20), (11.55, 57.30), (12.45, 55.20),
    ),
)

# The Eastern Europe profile intentionally uses a set of conservative country
# envelopes rather than pretending a coarse polygon is a political border. It is
# an architectural-style hint only. Envelopes cover the Baltic, Central-Eastern,
# Balkan, and western Black Sea countries commonly requested for OFP/CWA maps.
_EASTERN_EUROPE = RegionProfile(
    identifier="eastern_europe",
    display_name="Eastern Europe",
    country_aliases=frozenset({
        "pl", "pol", "poland", "polska",
        "cz", "cze", "czechia", "czech republic", "cesko", "česko",
        "sk", "svk", "slovakia", "slovensko",
        "hu", "hun", "hungary", "magyarorszag", "magyarország",
        "ro", "rou", "romania", "românia",
        "bg", "bgr", "bulgaria", "българия",
        "md", "mda", "moldova",
        "ua", "ukr", "ukraine", "україна",
        "by", "blr", "belarus", "беларусь",
        "lt", "ltu", "lithuania", "lietuva",
        "lv", "lva", "latvia", "latvija",
        "ee", "est", "estonia", "eesti",
        "si", "svn", "slovenia", "slovenija",
        "hr", "hrv", "croatia", "hrvatska",
        "ba", "bih", "bosnia and herzegovina", "bosna i hercegovina",
        "rs", "srb", "serbia", "srbija",
        "me", "mne", "montenegro", "crna gora",
        "mk", "mkd", "north macedonia", "macedonia",
        "al", "alb", "albania", "shqipëria", "shqiperia",
        "xk", "kosovo",
        "ru", "rus", "russia", "россия",
    }),
    envelopes_lon_lat=(
        # west, south, east, north
        (14.10, 49.00, 24.20, 54.90),  # Poland
        (12.00, 48.50, 18.90, 51.10),  # Czechia
        (16.80, 47.70, 22.60, 49.70),  # Slovakia
        (16.10, 45.70, 22.90, 48.60),  # Hungary
        (20.20, 43.60, 29.80, 48.30),  # Romania
        (22.30, 41.20, 28.70, 44.30),  # Bulgaria
        (26.60, 45.40, 30.20, 48.50),  # Moldova
        (22.10, 44.30, 40.20, 52.40),  # Ukraine
        (23.10, 51.20, 32.80, 56.20),  # Belarus
        (20.90, 53.80, 26.90, 56.50),  # Lithuania
        (20.80, 55.60, 28.30, 58.10),  # Latvia
        (21.50, 57.40, 28.20, 59.80),  # Estonia
        (13.30, 45.40, 16.70, 46.90),  # Slovenia
        (13.40, 42.30, 19.50, 46.60),  # Croatia
        (15.70, 42.50, 19.70, 45.30),  # Bosnia and Herzegovina
        (18.80, 42.20, 23.00, 46.20),  # Serbia
        (18.40, 41.80, 20.40, 43.60),  # Montenegro
        (20.40, 40.80, 23.10, 42.40),  # North Macedonia
        (19.20, 39.60, 21.10, 42.70),  # Albania
        (20.00, 41.80, 21.80, 43.30),  # Kosovo
        (27.50, 49.00, 45.00, 60.00),  # western European Russia
    ),
)

# Western Europe uses a separate procedural façade catalogue.  The envelopes
# are deliberately split around the places where a broad rectangle would spill
# into the Eastern Europe profile; explicit country tags remain authoritative.
_WESTERN_EUROPE = RegionProfile(
    identifier="western_europe",
    display_name="Western Europe",
    country_aliases=frozenset({
        "ad", "and", "andorra",
        "at", "aut", "austria", "österreich", "osterreich",
        "be", "bel", "belgium", "belgië", "belgique",
        "ch", "che", "switzerland", "schweiz", "suisse", "svizzera",
        "cy", "cyp", "cyprus",
        "de", "deu", "germany", "deutschland",
        "dk", "dnk", "denmark", "danmark",
        "es", "esp", "spain", "españa", "espana",
        "fi", "fin", "finland", "suomi",
        "fr", "fra", "france",
        "gb", "gbr", "uk", "united kingdom", "great britain",
        "gr", "grc", "greece", "hellas", "ελλάδα",
        "ie", "irl", "ireland", "éire", "eire",
        "is", "isl", "iceland", "ísland", "island",
        "it", "ita", "italy", "italia",
        "li", "lie", "liechtenstein",
        "lu", "lux", "luxembourg", "luxemburg",
        "mc", "mco", "monaco",
        "mt", "mlt", "malta",
        "nl", "nld", "netherlands", "nederland", "holland",
        "no", "nor", "norway", "norge",
        "pt", "prt", "portugal",
        "sm", "smr", "san marino",
        "va", "vat", "vatican city", "vatican",
    }),
    envelopes_lon_lat=(
        (-11.00, 49.50, 2.20, 61.00),   # Britain and Ireland
        (-25.00, 63.00, -12.00, 67.50), # Iceland
        (-10.00, 35.50, 4.50, 44.50),   # Iberia
        (-5.50, 42.00, 11.90, 55.50),   # France, Benelux, west Germany, Switzerland
        (11.90, 51.15, 14.10, 55.10),   # north-east Germany
        (11.90, 47.20, 15.10, 48.45),   # Bavaria and western Austria
        (6.00, 36.00, 19.00, 42.25),    # central and southern Italy
        (6.00, 42.25, 13.25, 47.50),    # northern Italy, clear of Slovenia/Croatia
        (7.50, 54.50, 15.50, 58.00),    # Denmark
        (-10.80, 35.50, 3.60, 43.90),   # Portugal and western Spain
    ),
)

# Middle Eastern palettes are selected before the broad Africa profile so Egypt
# and the Arabian/Levantine overlap use the arid masonry catalogue rather than
# the pan-African fallback. As with every profile here, these are style hints,
# not a political or cultural boundary dataset.
_MIDDLE_EAST = RegionProfile(
    identifier="middle_east",
    display_name="Middle East",
    country_aliases=frozenset({
        "bh", "bhr", "bahrain",
        "eg", "egy", "egypt", "مصر",
        "ir", "irn", "iran", "iran, islamic republic of", "ایران",
        "iq", "irq", "iraq", "العراق",
        "il", "isr", "israel", "ישראל",
        "jo", "jor", "jordan", "الأردن",
        "kw", "kwt", "kuwait", "الكويت",
        "lb", "lbn", "lebanon", "لبنان",
        "om", "omn", "oman", "عمان",
        "ps", "pse", "palestine", "state of palestine", "فلسطين",
        "qa", "qat", "qatar", "قطر",
        "sa", "sau", "saudi arabia", "السعودية",
        "sy", "syr", "syria", "syrian arab republic", "سوريا",
        "tr", "tur", "turkey", "türkiye", "turkiye",
        "ae", "are", "united arab emirates", "uae", "الإمارات",
        "ye", "yem", "yemen", "اليمن",
    }),
    envelopes_lon_lat=(
        (24.00, 21.00, 37.50, 32.20),  # Egypt and Sinai
        (25.00, 34.50, 45.50, 42.50),  # Turkey and northern Mesopotamia
        (34.00, 28.50, 43.00, 38.00),  # Levant and Iraq west
        (38.00, 24.00, 63.50, 40.50),  # Iraq, Iran and Gulf north
        (42.00, 12.00, 60.50, 31.50),  # Arabian Peninsula
    ),
)

# A continental profile is necessarily broad. Explicit country tags take
# precedence; the coarse outline only provides a useful default for source
# bundles that contain no country metadata. Madagascar is a separate envelope.
_AFRICA = RegionProfile(
    identifier="africa",
    display_name="Africa",
    country_aliases=frozenset({
        "dz", "dza", "algeria", "algérie",
        "ao", "ago", "angola",
        "bj", "ben", "benin", "bénin",
        "bw", "bwa", "botswana",
        "bf", "bfa", "burkina faso",
        "bi", "bdi", "burundi",
        "cv", "cpv", "cape verde", "cabo verde",
        "cm", "cmr", "cameroon", "cameroun",
        "cf", "caf", "central african republic",
        "td", "tcd", "chad", "tchad",
        "km", "com", "comoros", "comores",
        "cg", "cog", "republic of the congo", "congo-brazzaville",
        "cd", "cod", "democratic republic of the congo", "congo-kinshasa", "drc",
        "ci", "civ", "ivory coast", "côte d'ivoire", "cote d'ivoire",
        "dj", "dji", "djibouti",
        "gq", "gnq", "equatorial guinea",
        "er", "eri", "eritrea",
        "sz", "swz", "eswatini", "swaziland",
        "et", "eth", "ethiopia",
        "ga", "gab", "gabon",
        "gm", "gmb", "gambia", "the gambia",
        "gh", "gha", "ghana",
        "gn", "gin", "guinea",
        "gw", "gnb", "guinea-bissau",
        "ke", "ken", "kenya",
        "ls", "lso", "lesotho",
        "lr", "lbr", "liberia",
        "ly", "lby", "libya",
        "mg", "mdg", "madagascar",
        "mw", "mwi", "malawi",
        "ml", "mli", "mali",
        "mr", "mrt", "mauritania",
        "mu", "mus", "mauritius",
        "ma", "mar", "morocco", "maroc",
        "mz", "moz", "mozambique",
        "na", "nam", "namibia",
        "ne", "ner", "niger",
        "ng", "nga", "nigeria",
        "rw", "rwa", "rwanda",
        "st", "stp", "sao tome and principe", "são tomé and príncipe",
        "sn", "sen", "senegal", "sénégal",
        "sc", "syc", "seychelles",
        "sl", "sle", "sierra leone",
        "so", "som", "somalia",
        "za", "zaf", "south africa",
        "ss", "ssd", "south sudan",
        "sd", "sdn", "sudan",
        "tz", "tza", "tanzania", "united republic of tanzania",
        "tg", "tgo", "togo",
        "tn", "tun", "tunisia", "tunisie",
        "ug", "uga", "uganda",
        "zm", "zmb", "zambia",
        "zw", "zwe", "zimbabwe",
    }),
    polygon_lon_lat=(
        (-17.80, 37.20), (-5.00, 36.00), (10.00, 37.20), (24.00, 33.50),
        (34.50, 31.50), (37.50, 22.00), (43.50, 12.00), (51.50, 11.50),
        (50.00, 2.00), (42.00, -12.00), (36.00, -25.00), (28.00, -35.20),
        (17.00, -34.80), (11.00, -18.00), (5.00, -5.00), (-5.00, 4.00),
        (-17.50, 14.50), (-17.80, 37.20),
    ),
    envelopes_lon_lat=((43.00, -26.00, 51.00, -11.00),),  # Madagascar
)

REGION_PROFILES: tuple[RegionProfile, ...] = (
    _SWEDEN, _WESTERN_EUROPE, _EASTERN_EUROPE, _MIDDLE_EAST, _AFRICA
)


def _point_in_polygon(lon: float, lat: float, polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def detect_region(
    bbox: tuple[float, float, float, float],
    tag_sources: Sequence[Mapping[str, str]] = (),
) -> RegionProfile | None:
    """Detect the current architectural region.

    Explicit ISO country tags win. Otherwise the centre of the selected source
    bbox is tested against the small built-in region catalogue.
    """

    explicit_country_seen = False
    for tags in tag_sources:
        country = str(tags.get("addr:country") or tags.get("country_code") or "").casefold().strip()
        if not country:
            continue
        explicit_country_seen = True
        for profile in REGION_PROFILES:
            if country in profile.country_aliases:
                return profile
    if explicit_country_seen:
        return None
    south, west, north, east = bbox
    latitude = (south + north) * 0.5
    longitude = (west + east) * 0.5
    for profile in REGION_PROFILES:
        if profile.polygon_lon_lat and _point_in_polygon(
            longitude, latitude, profile.polygon_lon_lat
        ):
            return profile
        for minimum_lon, minimum_lat, maximum_lon, maximum_lat in profile.envelopes_lon_lat:
            if minimum_lon <= longitude <= maximum_lon and minimum_lat <= latitude <= maximum_lat:
                return profile
    return None
