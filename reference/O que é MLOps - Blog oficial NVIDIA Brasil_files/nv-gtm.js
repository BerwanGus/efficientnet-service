function nvEventTracker(category, action, opt_label, opt_value, opt_noninteractive) {
	try {
		dataLayer.push({
			'event': 'analyticsEvent',
			'eventCategory': category,
			'eventAction': action,
			'eventLabel': opt_label,
			'eventValue': opt_value,
			'eventNonInt': opt_noninteractive
		});
	} catch (e) {
	}
}

function nvBannerTracker(action, label, bannerType, bannerRegion, bannerPosition, bannerImageURL, bannerLandingURL, bannerName, bannerPageURL, bannerImpressions, bannerClicks) {

	try {
		dataLayer.push({
			'event': 'bannerTrackEvent',
			'eventCategory': 'Banner|' + bannerType,
			'eventAction': action,
			'eventLabel': label,
			'eventValue': '0',
			'eventNonInt': 'TRUE',
			'bannerName': bannerName,
			'bannerType': bannerType,
			'bannerRegion': bannerRegion,
			'bannerPosition': bannerPosition,
			'bannerImageURL': bannerImageURL,
			'bannerLandingURL': bannerLandingURL,
			'bannerPageURL': bannerPageURL,
			'bannerImpressions': bannerImpressions,
			'bannerClicks': bannerClicks
		});
	} catch (e) {
	}
}



